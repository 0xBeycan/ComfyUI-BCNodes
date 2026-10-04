# Text

## `BC_PromptList` — Prompt List

`prepend_text`, `multiline_text`, `append_text`, `start_index`, `max_rows` → `prompt` (list), `body_text` (list), `show_help` (string). One entry per line of `multiline_text`, windowed by `start_index` / `max_rows`; `prompt` wraps each line in the prepend / append texts, `body_text` is the bare line. Both lists are `OUTPUT_IS_LIST`, so downstream nodes run once per line.

## `BC_AspectPromptList` — Aspect Prompt List

`prepend_text`, `multiline_text`, `append_text` → `prompt` (list), `body_text` (list), `width` (`INT` list), `height` (`INT` list). `multiline_text` holds `[WxH]` header lines (e.g. `[1024x1536]`) with one prompt per line under each; blank lines are skipped. Every prompt gets the size of the header above it, so the four lists have the same length and index `i` of each belongs to the same prompt. Wire `width` / `height` into a latent node (e.g. `EmptySD3LatentImage`): one sampler chain then renders every prompt at its own size, once per line. A prompt line above the first header, a `0` side, or a text with no prompt lines is an error.

```
[1280x1280]
first square prompt
second square prompt

[1024x1536]
first portrait prompt
```

## `BC_ShowText` — Show Text

`text` (`STRING`, input only) → `STRING`. Shows the text in a read-only, growing box on the node and passes it on. A list input shows one box per element and is passed on as a list. The shown text is also written into the workflow metadata of saved images.
