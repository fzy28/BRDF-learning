import mitsuba as mi
import sys
import os
import time
from learned_brdf import MyBSDF
from tqdm import tqdm
mi.set_variant("cuda_ad_rgb")

start_time = time.time()

mi.register_bsdf("mybsdf", lambda props: MyBSDF(props))
# scene = mi.load_file("./disney_bsdf_test/disney_diffuse.xml")

scene = mi.load_file("./matpreview/scene.xml")
params = mi.traverse(scene)
# print(params)
SPP = 1
spp = SPP * 512

seed = 0
image = mi.render(scene, spp=SPP, seed=seed).numpy()
print(image.shape)
for _ in tqdm(range(spp // SPP)):
    image += mi.render(scene, spp=SPP, seed=seed).numpy()
    seed += 1
image /= (spp // SPP) + 1

mi.util.write_bitmap("my_first_render.png", image)
end_time = time.time()

print("Render time: " + str(end_time - start_time) + " seconds")