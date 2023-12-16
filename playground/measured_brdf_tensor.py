# This file contains the implementation of a BSDF that is based on a measured BRDF.
# Used for mitsuba renderer.

import mitsuba as mi
import drjit as dr
from utils import *
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from model import *

mi.set_variant("cuda_ad_rgb")
dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BRDF_SAMPLING_RES_THETA_H = 90
BRDF_SAMPLING_RES_THETA_D = 90
BRDF_SAMPLING_RES_PHI_D = 360


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

def create_local_shading_frame(normals):

    abs_normals = torch.abs(normals)
    tangents = torch.zeros_like(normals)
    
    # Conditions for selecting the tangent vector
    condition_x = (abs_normals[:, 0] <= abs_normals[:, 1]) & (abs_normals[:, 0] <= abs_normals[:, 2])
    condition_y = (abs_normals[:, 1] <= abs_normals[:, 0]) & (abs_normals[:, 1] <= abs_normals[:, 2])

    # Calculate tangent vectors based on the conditions
    tangents[condition_x] = torch.stack([-normals[condition_x, 1], normals[condition_x, 0], torch.zeros_like(normals[condition_x, 0])], dim=1)
    tangents[~condition_x & condition_y] = torch.stack([torch.zeros_like(normals[~condition_x & condition_y, 0]), -normals[~condition_x & condition_y, 2], normals[~condition_x & condition_y, 1]], dim=1)
    tangents[~condition_x & ~condition_y] = torch.stack([-normals[~condition_x & ~condition_y, 2], torch.zeros_like(normals[~condition_x & ~condition_y, 1]), normals[~condition_x & ~condition_y, 0]], dim=1)
    tangents = torch.nn.functional.normalize(tangents, dim=-1)

    binormal = torch.cross(tangents, normals,dim=-1)
    return normals, binormal, tangents

def std_to_half_diff(wi, wo):
    half = dr.normalize((wi + wo) / 2).torch()
    theta_h = torch.acos(half[..., 2])
    phi_h = torch.atan2(half[..., 1], half[..., 0])
    bi_normal = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)

    tmp = rotate_vector_tensor2(half, normal, -phi_h)
    tmp = tmp / torch.norm(tmp, dim=1, keepdim=True)
    diff = rotate_vector_tensor2(tmp, bi_normal, -theta_h)
    diff = diff / torch.norm(diff, dim=1, keepdim=True)
    
    
    theta_d = torch.acos(diff[..., 2])
    phi_d = torch.atan2(diff[..., 1], diff[..., 0])
    phi_d += torch.where(phi_d < 0, torch.pi, 0)
    return theta_h, theta_d, phi_d

def my_croodinate_system(n):
    sign = dr.sign(n[2])
    a = -1 / (sign + n[2])
    b = n[0] * n[1] * a
    
    s = mi.Vector3f(dr.mulsign(dr.sqr(n[0])*a, n[2])+1
                    ,dr.mulsign(b, n[2]),dr.mulsign(-n[0], n[2]))
    dr.fma
    t = mi.Vector3f(b,n[1]*n[1]*a+sign,-n[1])
    return s,t
def my_croodinate_system_torch(n):
    
    sign =torch.sign(n[..., 2])
    n[..., 2] = torch.where(torch.abs(n[..., 2])  == 0,  1e-10, n[..., 2])
    
    a = -1 / (sign + n[..., 2]) 
    
    b = torch.mul(n[..., 0], n[..., 1]) * a
    
    s1 = torch.square(n[..., 0])* a * torch.sign(n[..., 2])+1
    s2 = b * torch.sign(n[..., 2])
    s3 = -n[..., 0] * torch.sign(n[..., 2])
    
    s = torch.stack([s1,s2,s3],dim=-1)
    
    t1 = b
    t2 = n[..., 1]* n[..., 1]*a + sign
    t3 = -n[..., 1]
    t = torch.stack([t1,t2,t3],dim=-1)
    return s,t
def std_to_half_diff_world(wi, wo, n):
    

    s,t = my_croodinate_system_torch(n.torch())
    
    
    # s, t = my_croodinate_system_torch(n.torch())
    # wi,wo,n = wi.torch(),wo.torch(),n.torch()
    wilocal = torch.stack(
        [
            torch.sum(wi.torch() * s, dim=-1),
            torch.sum(wi.torch() * t, dim=-1),
            torch.sum(wi.torch() * n.torch(), dim=-1),
        ],
        dim=-1,
    )
    wilocal = wilocal / torch.norm(wilocal, dim=1, keepdim=True)
    wolocal = torch.stack(
        [
            torch.sum(wo.torch() * s, dim=-1),
            torch.sum(wo.torch() * t, dim=-1),
            torch.sum(wo.torch() * n.torch(), dim=-1),
        ],
        dim=-1,
    )

    wolocal = wolocal / torch.norm(wolocal, dim=1, keepdim=True)
    half = (wilocal + wolocal) / 2
    half = half / torch.norm(half, dim=1, keepdim=True)

    theta_h = torch.acos(half[..., 2])
    phi_h = torch.atan2(half[..., 1], half[..., 0])
    
    bi_normal = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    
    tmp = rotate_vector_tensor2(half, normal, -phi_h)
    tmp = tmp / torch.norm(tmp, dim=1, keepdim=True)
    diff = rotate_vector_tensor2(tmp, bi_normal, -theta_h)
    diff = diff / torch.norm(diff, dim=1, keepdim=True)
    
    theta_d = torch.acos(diff[..., 2])
    phi_d = torch.atan2(diff[..., 1], diff[..., 0])
    phi_d += torch.where(phi_d < 0, torch.pi, 0)
    return theta_h, theta_d, phi_d

MeasuredBRDF_global = None


class MyBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.filename = props["filename"]
        global MeasuredBRDF_global
        if MeasuredBRDF_global is None:
            MeasuredBRDF_global = MeasuredBRDF(self.filename).get_array()
            MeasuredBRDF_global = torch.from_numpy(MeasuredBRDF_global).float().to(device)
        # Set the BSDF flags
        reflection_flags = (
            mi.BSDFFlags.DiffuseReflection 
            | mi.BSDFFlags.FrontSide
        )
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags
        
    def half_diff_look_up_brdf(self, theta_h, theta_d, phi_d):
        
        theta_half_deg = theta_h / (torch.pi * 0.5) * BRDF_SAMPLING_RES_THETA_H
        
        id_theta_h = torch.clip(
            torch.sqrt(theta_half_deg * BRDF_SAMPLING_RES_THETA_H),
            0,
            BRDF_SAMPLING_RES_THETA_H - 1,
        )
        id_theta_d = torch.clip(
            theta_d / (torch.pi * 0.5) * BRDF_SAMPLING_RES_THETA_D,
            0,
            BRDF_SAMPLING_RES_THETA_D - 1,
        )
        id_phi_d = torch.clip(
            phi_d / torch.pi * BRDF_SAMPLING_RES_PHI_D / 2,
            0,
            BRDF_SAMPLING_RES_PHI_D // 2 - 1,
        )
        
        # return the value with nearest index value, officially used in the Merl BRDF
        id_theta_h = (id_theta_h).int()
        id_theta_d = (id_theta_d).int()
        id_phi_d = (id_phi_d).int()
        # print(id_theta_d, id_phi_d, id_theta_h)
        return torch.clip(MeasuredBRDF_global[id_theta_h, id_theta_d, id_phi_d, :], 0, None)
    
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
        
        # wi = si.to_world(si.wi)
        # wo = si.to_world(bs.wo)
        # n = si.n
        # s,t =  mi.coordinate_system(n)
    
        # wilocal = mi.Vector3f(dr.dot(wi, s), dr.dot(wi, t), dr.dot(wi, n))
        # wolocal = mi.Vector3f(dr.dot(wo, s), dr.dot(wo, t), dr.dot(wo, n))

        value = self.half_diff_look_up_brdf(theta_h, theta_d, phi_d)
        value = mi.Vector3f(value[..., 0], value[..., 1], value[..., 2]) 

        return (bs, dr.select(active & (bs.pdf > 0.0), value, mi.Vector3f(0)))

    def eval(self, ctx, si, wo, active=True):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        wi = si.wi

        theta_h, theta_d, phi_d = std_to_half_diff(wi, wo)

        # wi = si.to_world(si.wi)
        # wo = si.to_world(wo)
        # theta_h, theta_d, phi_d = std_to_half_diff_world(wi, wo, si.n)
        # theta_h = torch.clip(theta_h, 0, np.pi / 2)
        # theta_d = torch.clip(theta_d, 0, np.pi / 2)
        
        
        value = self.half_diff_look_up_brdf(theta_h, theta_d, phi_d)
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
    SPP = 16
    spp = SPP * 256

    seed = 0
    image = mi.render(scene, spp=SPP, seed=seed).numpy()
    print(image.shape)
    for _ in tqdm(range(spp // SPP)):
        image += mi.render(scene, spp=SPP, seed=seed).numpy()
        seed += 1
    image /= (spp // SPP) + 1

    mi.util.write_bitmap("cuda_measured2.png", image)
    end_time = time.time()

    print("Render time: " + str(end_time - start_time) + " seconds")
