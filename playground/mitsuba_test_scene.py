import mitsuba as mi
import sys
import os
import time
start_time = time.time()
mi.set_variant('cuda_ad_rgb')

scene = mi.load_file("./matpreview/matpreview.xml")

image = mi.render(scene, spp=256)
mi.util.write_bitmap("my_first_render4.png", image)
end_time = time.time()

print("Render time: " + str(end_time - start_time) + " seconds")