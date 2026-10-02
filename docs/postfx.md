# PostFx

Half-precision input (float16 or bfloat16; BCVideoNodes' Load Video gives float16 by default): PostFx Apply reads it a frame at a time, and PostFx Signature Sheet only its first frame, as the float32 8-bit levels a float32 load holds; postfx works in float32 and the output takes the input's dtype; any other input gives float32 outputs, as before.

## `BC_PostFxApply` — PostFx Apply, and the look nodes

Film-emulation looks from the [`postfx`](https://github.com/0xBeycan/postfx) pipeline: film stocks, cinematic grades and `.cube` LUTs defined in YAML, running on the CPU. Pick a look, dial a shooting condition and a global strength, and every image gets the same reusable visual identity.

| Node | In → Out | What it does |
| --- | --- | --- |
| PostFx Apply | `IMAGE` (+ `look`, `mask`) → `IMAGE` | The core node. Applies a **theme** + **condition** + **strength** to an image batch. `theme = none` passes the image through untouched. A connected `look` overrides the theme dropdown; an optional `mask` limits the effect to the masked region (an all-black mask is ignored). |
| PostFx Theme | → `POSTFX_LOOK` | Emits a built-in theme as a look, to start a chain from a named theme. |
| PostFx Custom Look | (`look`) → `POSTFX_LOOK` | Builds a look from common controls (white balance, exposure, contrast, vibrance / saturation, grain, vignette, halation, clarity). With a `look` input, only the knobs moved off neutral override it. |
| PostFx LUT | (`look`) → `POSTFX_LOOK` | Attaches a 3D `.cube` LUT. Standalone by default; connect a `look` to layer the LUT on top of a theme. |
| PostFx Signature Sheet | `IMAGE` → `IMAGE` | Labeled contact-sheet grid of every theme in a category, for side-by-side comparison. |

`POSTFX_LOOK` is the link type between the look-producing nodes and PostFx Apply; every look node has an optional `look` input, so they chain in any order:

```
PostFx Theme (portra_400) → PostFx LUT (my_look.cube) → PostFx Custom Look (grain ↑) → PostFx Apply (condition = neon_night) → Save Image
```

- **Theme** = the look (colour / grain / lens): 3 texture-only `grain` finishes (fine per-pixel grain, colour untouched — the default social-still finish; chain a signature theme before one for a look), 15 `signature` film stocks and industry grades, 15 `experimental`.
- **Condition** scales only the texture (grain, chroma noise, halation) for the shooting situation — `neutral`, `day_outdoor`, `overcast`, `indoor_evening`, `neon_night`, `night_flash` — and leaves colour alone.
- **Strength** `0–1.5` blends the whole effect with the original. **Seed** makes grain deterministic; `batch_seed = increment` gives each frame of a batch its own grain.
- **LUTs**: drop `.cube` 3D LUTs into this repo's [`luts/`](../luts/) folder to see them in the PostFx LUT dropdown, or point `lut_path` at any absolute path. A LUT applies mid-pipeline, so a theme's grade runs before it and grain / vignette / sharpen finish on top.
