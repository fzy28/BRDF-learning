# Used for mitsuba renderer.
# A simple diffuse brdf for testing.
import mitsuba as mi
import drjit as dr
from utils import *
mi.set_variant("cuda_ad_rgb")
import time

class MyBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.filename = props["filename"]
        self.alpha = 0.01
        self.albedo = mi.Color3f([0.82 ,0.67 ,0.16]) / 2.2
        print(self.albedo)
        self.albedo = mi.Color3f(self.albedo)
        print(self.albedo)
        # Set the BSDF flags
        reflection_flags = (
            mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide 
        )
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags

    def sample(self, ctx, si, sample1, sample2, active=True):
        # Compute Fresnel terms
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        active &= cos_theta_i > 0

        
        bs = mi.BSDFSample3f()
        bs.wo = mi.warp.square_to_beckmann(sample2, self.alpha)
        bs.pdf = mi.warp.square_to_beckmann_pdf(bs.wo, self.alpha)
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+self.m_flags)
        bs.sampled_component = 0

        value = self.albedo * dr.inv_pi 

        
        return (bs, dr.select(active & (bs.pdf > 0.0), value, mi.Vector3f(0)))

    def eval(self, ctx, si, wo, active=True):
    
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        value = self.albedo

        value = value * dr.inv_pi 

        return dr.select(
            (cos_theta_i > 0.0) & (cos_theta_o > 0.0), value, mi.Vector3f(0)
        )

    def pdf(self, ctx, si, wo, active=True):


        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        pdf = mi.warp.square_to_beckmann_pdf(wo,self.alpha)

        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def traverse(self, callback):
        return

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        return "MyBSDF[\n" "    albedo=%s,\n" "]" % (self.albedo)


if __name__ == "__main__":
    start_time = time.time()

    mi.register_bsdf("mybsdf", lambda props: MyBSDF(props))
    #scene = mi.load_file("./disney_bsdf_test/disney_diffuse.xml")

    scene = mi.load_file("./matpreview/scene.xml")
    #params = mi.traverse(scene)
    #print(params)
    image = mi.render(scene, spp=256) 
    
    mi.util.write_bitmap("my_first_render4.png", image)
    end_time = time.time()

    print("Render time: " + str(end_time - start_time) + " seconds")