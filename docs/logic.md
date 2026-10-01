# Logic

## `BC_LogicBoolean` — Logic Boolean

A `FLOAT` widget in `[0, 1]` (default `1`) is rounded to a boolean.

| Output | Value for widget `0.7` |
| --- | --- |
| `BOOLEAN` | `True` |
| `NUMBER` | `1` |
| `INT` | `1` |
| `FLOAT` | `0.7` (unrounded) |

## `BC_MathExpression` — Math Expression

`expression` (multiline) plus optional `a`, `b`, `c` (`INT`, `FLOAT`, `IMAGE` or `LATENT`) → `INT`, `FLOAT` (the same value, truncated and as a float). The result is also drawn on the node. Re-evaluated on every run.

The expression is parsed with `ast` and walked with a whitelist — there is no `eval()`. Allowed: numbers; `+ - * / // % **`, `& | ^ << >>`; unary `- + ~ not`; `and` / `or`; comparisons (`== != < <= > >=`, chained) which yield `1` / `0`; the names `a`, `b`, `c`; `a.width` / `a.height` for an `IMAGE` or `LATENT` input (latent sizes are multiplied by 8); `NodeTitle.widget` to read another node's widget by node title or type; and the functions `min max abs round int float pow sqrt floor ceil randomint randomchoice iif`. Anything else — subscripts, strings, lambdas, other attributes or functions — is a `ValueError` naming the offending piece. An empty expression evaluates to `0`; a referenced input that is not connected is an error.

## `BC_AnySwitch` — Any Switch

Wildcard inputs `any_01`, `any_02`, … → the first one that is connected and not `None`. Nothing connected → `None`. Slots grow as they are connected (one empty slot always waits at the end; empty slots in the middle are removed and the rest renumbered), and the socket type follows whatever is connected so the canvas shows and checks the real type. Useful with an optional branch: wire the optional source first and a fallback second.

## `BC_SelectSwitch` — Select Switch

Named options, one wildcard input each, plus `selected` (a combo of the option names) → the input of the selected option. Like ComfyUI's boolean Switch, but chosen by name and with any number of options. `+ Add option` asks for a name; each option is a row under the combo: click its name to rename it (the link stays), `✕` to remove it. Empty, duplicate and reserved names (`selected`, `self`) are refused. The option list is saved with the workflow. The socket type follows whatever is connected, as in Any Switch.

The option inputs are lazy: only the selected option's branch is executed, so an image can feed two expensive preprocessors and only the selected one runs. The selected option having no input connected is an error naming the option. `selected` can be promoted out of a subgraph or driven by a `STRING` / `COMBO` output.

## `BC_Seed` — Seed

`seed` (`INT`, `-1 … 0xffffffffffffffff`, default `0`) → `SEED`. Three buttons under the widget:

| Button | Effect |
| --- | --- |
| `🎲 Randomize Each Time` | Sets the widget to `-1`: every queued run gets a fresh random seed. |
| `🎲 New Fixed Random` | Writes a concrete random seed into the widget; it then stays fixed. |
| `♻️ (Use Last Queued Seed)` | Puts the seed of the last queued run back into the widget. Greyed out and in parentheses until a run has used a seed that differs from the widget; after a random run it reads `♻️ <seed>`. |

`-1` never travels: right before a prompt is sent, the frontend replaces it with a drawn seed in the API prompt and in the workflow copy that ends up in image metadata, so dropping a saved image onto the canvas brings the real seed back. A prompt queued through the API with `-1` gets the same treatment on the server. Random seeds are drawn below 2⁵³ so they survive the round trip through JavaScript exactly. The node re-evaluates every run; downstream nodes re-run only when the seed actually changes.
