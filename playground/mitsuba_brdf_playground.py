import mitsuba as mi
import sys
import os
import time
import drjit as dr

from measured_brdf import MyBSDF
start_time = time.time()
mi.set_variant('llvm_ad_rgb')

bsdf = mi.load_dict({
    'type': 'roughconductor',
    'alpha': 0.1,
    'distribution': 'ggx'
})


def sph_to_dir(theta, phi):
    """Map spherical to Euclidean coordinates"""
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)

# Create a (dummy) surface interaction to use for the evaluation of the BSDF
si = dr.zeros(mi.SurfaceInteraction3f)

# Specify an incident direction with 45 degrees elevation
si.wi = sph_to_dir(dr.deg2rad(45.0), 0.0)

# Create grid in spherical coordinates and map it onto the sphere
res = 300
theta_o, phi_o = dr.meshgrid(
    dr.linspace(mi.Float, 0,     dr.pi,     res),
    dr.linspace(mi.Float, 0, 2 * dr.pi, 2 * res)
)
wo = sph_to_dir(theta_o, phi_o)

# Evaluate the whole array (18000 directions) at once
values = bsdf.eval(mi.BSDFContext(), si, wo)

import numpy as np
values_np = np.array(values)

import matplotlib.pyplot as plt

# Extract red channel of BRDF values and reshape into 2D grid
values_r = values_np[:, 0]
values_r = values_r.reshape(2 * res, res).T

# Plot values for spherical coordinates
fig, ax = plt.subplots(figsize=(8, 4))

im = ax.imshow(values_r, extent=[0, 2 * np.pi, np.pi, 0], cmap='jet')

ax.set_xlabel(r'$\phi_o$', size=10)
ax.set_xticks([0, dr.pi, dr.two_pi])
ax.set_xticklabels(['0', '$\\pi$', '$2\\pi$'])
ax.set_ylabel(r'$\theta_o$', size=10)
ax.set_yticks([0, dr.pi / 2, dr.pi])
ax.set_yticklabels(['0', '$\\pi/2$', '$\\pi$'])

plt.show()