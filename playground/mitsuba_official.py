import mitsuba as mi
import sys
import os
import time
from measured_brdf import MyBSDF
start_time = time.time()
mi.set_variant('llvm_ad_rgb')
mi.register_bsdf("mybsdf", lambda props: MyBSDF(props))
my_bsdf = mi.load_dict({
    'type' : 'mybsdf',
    'tint' : [0.2, 0.9, 0.2],
    'eta' : 1.33
})

scene = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': 'path'
    },
    'light': {
        'type': 'constant',
        'radiance': 0.99,
    },
    'sphere' : {
        'type': 'sphere',
        'bsdf': my_bsdf
    },
    'sensor': {
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f.look_at(origin=[0, -5, 5],
                                                 target=[0, 0, 0],
                                                 up=[0, 0, 1]),
    }
})

image = mi.render(scene)

mi.util.write_bitmap("my_first_render4.png", image)
end_time = time.time()

print("Render time: " + str(end_time - start_time) + " seconds")