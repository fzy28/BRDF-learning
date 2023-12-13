# This file contains the implementation of a BSDF that is based on a measured BRDF.
# Used for mitsuba renderer.

import mitsuba as mi
import drjit as dr
from utils import *
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from model import *

mi.set_variant("cuda_ad_rgb")
dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rotate_vector(vector, axis, angle):
    angle = angle[:, np.newaxis]
    rotation = R.from_rotvec(angle * np.array(axis))
    rotated_vec = rotation.apply(vector)
    return rotated_vec


def rotate_vector_tensor(vector, axis, angle):
    # not implemented yet
    vector = vector.detach().cpu().numpy()
    axis = axis.detach().cpu().numpy()
    angle = angle.detach().cpu().numpy()
    result = rotate_vector(vector, axis, angle)
    return torch.from_numpy(result).float().to(device)


def rotate_vector_tensor2(vector, axis, angle):
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]],device=device)
    K = K.unsqueeze(0).repeat(vector.shape[0], 1, 1)
    I = torch.eye(3,device=device).unsqueeze(0).repeat(vector.shape[0], 1, 1)
    R = I + torch.sin(angle).view(-1, 1, 1) * K + (1 - torch.cos(angle).view(-1, 1, 1)) * torch.matmul(K, K)

    return torch.bmm(R, vector.unsqueeze(-1)).squeeze(-1)

def std_to_half_diff(wi, wo):
    half = dr.normalize(wi + wo).torch()
    theta_h = torch.arccos(half[..., 2])
    phi_h = torch.arctan2(half[..., 1], half[..., 0])
    bi_normal = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)

    tmp = rotate_vector_tensor2(half, normal, -phi_h)
    tmp = tmp / torch.norm(tmp, dim=1, keepdim=True)
    diff = rotate_vector_tensor2(tmp, bi_normal, -theta_h)
    diff = diff / torch.norm(tmp, dim=1, keepdim=True)
    theta_d = torch.arccos(half[..., 2])
    phi_d = torch.arctan2(half[..., 1], half[..., 0])
    return theta_h, theta_d, phi_d


model_global = None


class MyBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.filename = props["filename"]
        global model_global
        if model_global is None:
            model_global = NN().to(device)
            model_path = "be_simple_mlp_" + self.filename + ".pth"
            model_global.load_state_dict(torch.load(model_path))
            model_global.eval()
        # Set the BSDF flags
        reflection_flags = (
            mi.BSDFFlags.SpatiallyVarying
            | mi.BSDFFlags.DiffuseReflection
            | mi.BSDFFlags.FrontSide
        )
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags

    def sample(self, ctx, si, sample1, sample2, active=True):
        # Compute Fresnel terms

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        active &= cos_theta_i > 0

        bs = mi.BSDFSample3f()
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+self.m_flags)
        bs.sampled_component = 0

        wi = si.wi
        wo = bs.wo

        theta_h, theta_d, phi_d = std_to_half_diff(wi, wo)

        theta_h = torch.clip(theta_h, 0, np.pi / 2)
        theta_d = torch.clip(theta_d, 0, np.pi / 2)
        phi_d = torch.clip(phi_d, 0, np.pi)

        conca_in = torch.stack([theta_h, theta_d, phi_d], dim=1)

        value = model_global(conca_in)
        value = value
        value = mi.Vector3f(value[..., 0], value[..., 1], value[..., 2])

        return (bs, dr.select(active & (bs.pdf > 0.0), value, mi.Vector3f(0)))

    def eval(self, ctx, si, wo, active=True):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        wi = si.wi

        theta_h, theta_d, phi_d = std_to_half_diff(wi, wo)

        theta_h = torch.clip(theta_h, 0, np.pi / 2)
        theta_d = torch.clip(theta_d, 0, np.pi / 2)
        phi_d = torch.clip(phi_d, 0, np.pi)

        conca_in = torch.stack([theta_h, theta_d, phi_d], dim=1)

        value = model_global(conca_in)
        # value = value.detach().cpu().numpy()
        value = mi.Vector3f(value[..., 0], value[..., 1], value[..., 2])
        return dr.select(
            (cos_theta_i > 0.0) & (cos_theta_o > 0.0), value, mi.Vector3f(0)
        )

    def pdf(self, ctx, si, wo, active=True):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)

        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def traverse(self, callback):
        callback.put_parameter("LearnedBSDF", self, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        return "MyBSDF[\n" "    albedo=%s,\n" "]" % (self.albedo)


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    import time

    start_time = time.time()

    mi.register_bsdf("mybsdf", lambda props: MyBSDF(props))
    # scene = mi.load_file("./disney_bsdf_test/disney_diffuse.xml")

    scene = mi.load_file("./matpreview/scene.xml")
    params = mi.traverse(scene)
    # print(params)
    SPP = 1
    spp = SPP * 1024

    seed = 0
    image = mi.render(scene, spp=SPP, seed=seed).numpy()
    print(image.shape)
    for _ in tqdm(range(spp // SPP)):
        image += mi.render(scene, spp=SPP, seed=seed).numpy()
        seed += 1
    image /= (spp // SPP) + 1

    mi.util.write_bitmap("cuda_learned.png", image)
    end_time = time.time()

    print("Render time: " + str(end_time - start_time) + " seconds")
