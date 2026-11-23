"""Detailer: a region of an image sampled again at a larger size and pasted back, one module per
concern:

    region        the crop around a region and the size it is sampled at
    conditioning  every ControlNet's hint cut to the crop; core's inpaint conditioning of the crop
    core          the detailer core (Detailer's flow): crop -> two sampling passes -> paste
    face          Face Detailer's flow: SAM 3 faces, the character, the size gate, mouth and eye passes
"""
