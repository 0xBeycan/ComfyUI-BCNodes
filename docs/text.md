# Text

## `BC_PromptList` — Prompt List

`prepend_text`, `multiline_text`, `append_text`, `start_index`, `max_rows` → `prompt` (list), `body_text` (list), `show_help` (string). One entry per line of `multiline_text`, windowed by `start_index` / `max_rows`; `prompt` wraps each line in the prepend / append texts, `body_text` is the bare line. Both lists are `OUTPUT_IS_LIST`, so downstream nodes run once per line.

## `BC_ShowText` — Show Text

`text` (`STRING`, input only) → `STRING`. Shows the text in a read-only, growing box on the node and passes it on. A list input shows one box per element and is passed on as a list. The shown text is also written into the workflow metadata of saved images.
