# Drop your `.cube` LUTs here

Any `.cube` **3D LUT** file placed in this folder shows up in the **PostFx LUT**
node's `lut` dropdown. You can also point the node's `lut_path` field at any
absolute path instead.

- Only **3D** `.cube` LUTs are supported (the common creative-look format).
- The LUT is applied in display (sRGB) space, mid-pipeline — so a theme's color
  grade runs *before* it and grain/vignette/sharpen finish *on top*.
- No LUTs are bundled here (to avoid shipping third-party/licensed files). The
  `postfx` package ships one example (`punch_overlay`) selectable as a theme.
