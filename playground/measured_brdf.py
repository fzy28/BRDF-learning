# This file contains the implementation of a BSDF that is based on a measured BRDF.
# Used for mitsuba renderer.

import mitsuba as mi
import drjit as dr
from utils import *
class MyBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        # Load the measured BRDF data
        self.filename = props['filename']
        self.brdf = read_brdf(self.filename)


        # Set the BSDF flags
        reflection_flags   = mi.BSDFFlags.DeltaReflection   | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components  = [reflection_flags, transmission_flags]
        self.m_flags = reflection_flags | transmission_flags

    def sample(self, ctx, si, sample1, sample2, active):
        return 0.0, 0.0, 0.0, 0.0

    def eval(self, ctx, si, wo, active):
        return 0.0

    def pdf(self, ctx, si, wo, active):
        return 0.0

    def eval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    def traverse(self, callback):
        # No need to do differentation here
        return

    def parameters_changed(self, keys):
        return

    def to_string(self):
        return ('MyBSDF[\n'
                '    filename=%s,\n'
                ']' % (self.filename))