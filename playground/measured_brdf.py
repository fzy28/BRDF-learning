# This file contains the implementation of a BSDF that is based on a measured BRDF.
# Used for mitsuba renderer.

import mitsuba as mi
import drjit as dr
from utils import *
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

mi.set_variant("llvm_ad_rgb")
dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)

def rotate_vector(vector, axis, angle):
    angle = angle[:, np.newaxis]
    rotation = R.from_rotvec(angle * np.array(axis))
    rotated_vec = rotation.apply(vector)
    return rotated_vec
def std_to_half_diff(wi, wo):
    
    half = dr.normalize(wi + wo).numpy()
        
        
    theta_h = np.arccos(half[...,2])
    phi_h = np.arctan2(half[...,1], half[...,0])
    bi_normal = np.array([0, 1, 0])
    normal = np.array([0, 0, 1])
    
    tmp = rotate_vector(half, normal, -phi_h)
    tmp = tmp / np.linalg.norm(tmp,axis=1, keepdims=True)
    diff = rotate_vector(tmp, bi_normal, -theta_h)
    diff = diff / np.linalg.norm(diff,axis=1, keepdims=True)
    
    theta_d = np.arccos(half[...,2])
    phi_d = np.arctan2(half[...,1], half[...,0])
    
    return theta_h, theta_d, phi_d
MeasuredBRDF_global = None
class MyBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        global MeasuredBRDF_global
        self.filename = props["filename"]
        if MeasuredBRDF_global is None:
            MeasuredBRDF_global = MeasuredBRDF(self.filename)
        # Set the BSDF flags
        reflection_flags = (
             mi.BSDFFlags.DiffuseReflection
            | mi.BSDFFlags.FrontSide
        )
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags
    def sample(self, ctx, si, sample1, sample2, active=True):
        # Compute Fresnel terms
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        t = si.wi
        t1 = si.to_local(si.wi) 
        active &= cos_theta_i > 0

        bs = mi.BSDFSample3f()
        bs.wo = mi.warp.square_to_beckmann(sample2, 0.1)
        bs.pdf = mi.warp.square_to_beckmann_pdf(bs.wo, 0.1)
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+self.m_flags)
        bs.sampled_component = 0
        
        wi = si.wi
        wo = bs.wo       
        
        theta_h, theta_d, phi_d = std_to_half_diff(wi, wo)
        
        value = MeasuredBRDF_global.half_diff_look_up_brdf(theta_h, theta_d, phi_d)
        value = mi.Vector3f(value[...,0],value[...,1],value[...,2])

        

        return (bs, dr.select(active & (bs.pdf > 0.0), value, mi.Vector3f(0)))

    def eval(self, ctx, si, wo, active=True):
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        
        wi = si.wi
        
        theta_h, theta_d, phi_d = std_to_half_diff(wi, wo)
        value = MeasuredBRDF_global.half_diff_look_up_brdf(theta_h, theta_d, phi_d)
        
        value = mi.Vector3f(value[...,0],value[...,1],value[...,2])
        return 0.0
    

    def pdf(self, ctx, si, wo, active=True):

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        
        pdf = mi.warp.square_to_beckmann_pdf(wo, 0.1)

        return 0.0

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def traverse(self, callback):
        callback.put_parameter("MeasuredBSDF", self, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        return "MyBSDF[\n" "    albedo=%s,\n" "]" % (self.albedo)

if __name__ == "__main__":
    mi.set_variant("llvm_ad_rgb")
    import time
    
    start_time = time.time()

    mi.register_bsdf("mybsdf", lambda props: MyBSDF(props))
    #scene = mi.load_file("./disney_bsdf_test/disney_diffuse.xml")

    scene = mi.load_file("./matpreview/scene.xml")
    params = mi.traverse(scene)
    #print(params)
    SPP = 32
    spp = SPP * 16 
    
    seed = 0
    image = mi.render(scene,spp=SPP,seed=seed).numpy()
    print(image.shape)
    for _ in tqdm(range(spp//SPP)):
        image += mi.render(scene,spp=SPP,seed=seed).numpy()
        seed += 1
    image /= (spp//SPP) + 1
    
    
    mi.util.write_bitmap("llvm_measured.png", image)
    end_time = time.time()

    print("Render time: " + str(end_time - start_time) + " seconds")