# PostFx

Film-emulation looks from the [`postfx`](https://github.com/0xBeycan/postfx) package: film stocks, cinematic grades and `.cube` LUTs defined in YAML, running on the CPU. A look, a shooting condition and a strength are applied to every image of a batch.

The nodes need `postfx>=1.1.0`, in the pack's `requirements.txt`. Without it, PostFx Apply, PostFx Theme and PostFx Signature Sheet are missing from the node list, because their dropdowns come from postfx (the console logs the error), and PostFx Custom Look and PostFx LUT stay listed but stop with the install message (`pip install 'postfx>=1.1.0'`) when run.

Half-precision input (float16 or bfloat16, e.g. BCVideoNodes' Load Video or SeedVR2 PostProcess): PostFx Apply reads it a frame at a time, and PostFx Signature Sheet only its first frame, as float32; postfx works in float32 and the output takes the input's dtype; float32 input works as before. A float16 frame whose values all lie within 1/16 of an 8-bit level is 8-bit data and is read as those levels, exactly its float32 source; any other half frame as its exact values.

## Looks

`POSTFX_LOOK` is the link type between the look nodes (PostFx Theme, PostFx Custom Look, PostFx LUT) and PostFx Apply; every look node has an optional `look` input, so they chain in any order:

```
PostFx Theme (signature/01_portra_400) → PostFx LUT (my_look.cube) → PostFx Custom Look (grain ↑) → PostFx Apply (condition = neon_night) → Save Image
```

- **Theme** = the look (colour / grain / lens), from the installed postfx. The dropdowns list each theme as `category/stem`, e.g. `signature/01_portra_400`. The categories: `grain` (texture-only finishes: fine per-pixel grain, colour untouched; chain a signature theme before one for a look), `signature` (film stocks and industry grades), `luts` (themes built on a `.cube` LUT that postfx ships, e.g. `punch_overlay`) and `experimental`.
- **Condition** scales only the texture (grain, chroma noise, halation) for the shooting situation and leaves colour alone. postfx's conditions: `neutral`, `day_outdoor`, `overcast`, `indoor_evening`, `neon_night`, `night_flash`.
- **Strength** blends the whole effect with the original: `0` is the original, `1` the full look, above `1` past it.

## `BC_PostFxApply` — PostFx Apply

`image` (`IMAGE`) + optional `look` (`POSTFX_LOOK`) and `mask` (`MASK`) → `image` (`IMAGE`). Applies a theme, a condition and a strength to every frame of the batch.

| Widget | Default | Range | Meaning |
| --- | --- | --- | --- |
| `theme` | the `portra_400` theme (`signature/01_portra_400`) | `none` + every theme | The look. `none` with no `look` connected passes the image through untouched. Ignored when `look` is connected. |
| `condition` | `neutral` | the conditions | Scales grain, chroma noise and halation only. |
| `strength` | `1.0` | `0..1.5` | `0` original, `1` full look, above `1` past it. |
| `seed` | `0` | `0..0xffffffff` | Grain seed: the same seed gives the same grain. |
| `batch_seed` | `fixed` | `fixed`, `increment` | `increment` adds the frame index to `seed`, so each frame of a batch gets its own grain. |

A connected `look` overrides `theme`. A `mask` limits the effect to its white area: each frame is blended with the original through the mask, resized to the image. One mask serves every frame; a shorter mask batch repeats its last mask. An all-black mask (e.g. LoadImage's for an image without alpha) is ignored and the effect applies everywhere. The output is clipped to `0..1`.

## `BC_PostFxTheme` — PostFx Theme

`theme` (default the `portra_400` theme, `signature/01_portra_400`) → `look` (`POSTFX_LOOK`). A built-in theme as a look, to start a chain from a named theme (Theme → LUT / Custom Look → Apply). For a theme on its own, pick it on PostFx Apply instead.

## `BC_PostFxCustomLook` — PostFx Custom Look

Optional `look` (`POSTFX_LOOK`) → `look` (`POSTFX_LOOK`). Builds a look from common controls. Without a `look` input the controls go onto a neutral look (every op at its no-op value); with one, onto a copy of it. A control at its neutral value leaves that op of the base look as it is, so a base look plus one moved control changes only that op. `temp` and `tint` are written together, as are `vibrance` and `saturation`; `grain_size` only with `grain` above `0`.

| Widget | Default (neutral) | Range | Meaning |
| --- | --- | --- | --- |
| `temp` | `0` | `-1..1` | White balance: above `0` warm, below `0` cool. |
| `tint` | `0` | `-1..1` | Above `0` magenta, below `0` green. |
| `exposure` | `0` | `-3..3` | Stops of light. |
| `contrast` | `0` | `0..1` | S-curve contrast strength. |
| `vibrance` | `0` | `-1..1` | Lifts low-saturation pixels, protects skin. |
| `saturation` | `1` | `0..2` | Uniform saturation (`1` = unchanged). |
| `grain` | `0` | `0..0.1` | Luma grain amount. |
| `grain_size` | `1.5` | `0.5..4` | Grain size, in pixels at a 1024 px edge. |
| `vignette` | `0` | `0..1` | Edge darkening amount. |
| `halation` | `0` | `0..2` | Highlight bloom strength. |
| `clarity` | `0` | `-1..1` | Local contrast; below `0` softens. |

## `BC_PostFxLut` — PostFx LUT

`lut` (`none` + the `.cube` files in [`luts/`](../luts/), default `none`), `intensity` (`0..1`, default `1`: the LUT blended with its input, `0` = off), optional `lut_path` (`STRING`, a path to a `.cube` file; overrides `lut`) and `look` (`POSTFX_LOOK`) → `look` (`POSTFX_LOOK`).

Attaches a 3D `.cube` LUT to a look: standalone by default (on a neutral look), layered on top of a connected `look`, e.g. a theme. A LUT applies mid-pipeline, in display (sRGB) space, so a theme's grade runs before it and grain / vignette / sharpen finish on top. Only 3D LUTs are supported. `lut` at `none` with an empty `lut_path` passes the look on unchanged (a neutral look when none is connected); a file that does not exist stops the run (`LUT file not found`). Drop `.cube` files into `luts/` (ignored by git) to see them in the `lut` dropdown; see [`luts/README.md`](../luts/README.md).

## `BC_PostFxSignatureSheet` — PostFx Signature Sheet

`image` (`IMAGE`), `category` (`signature`, `experimental` or `all`; default `signature`), `condition` (default `neutral`), `strength` (`0..1.5`, default `1`), `columns` (`1..10`, default `5`) → `sheet` (`IMAGE`, one image).

A labelled contact sheet of the batch's first image with every theme of `category` applied (`all`: every category), for side-by-side comparison. postfx builds it: each tile is the image, scaled down to postfx's thumbnail width when wider, every tile gets the same grain seed, and each is labelled with its theme's stem.
