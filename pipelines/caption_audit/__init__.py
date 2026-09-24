"""Caption Audit: the audit plumbing (audit.py) and the fixed-size card (card.py).

A caption set is a folder, not a string: images plus same-named `.txt`
sidecars. So the node takes a path, runs the audit in-process (no subprocess,
no temp caption files) and draws the result as a preview image inside the
node. `caption_audit` is driven entirely through its public surface:

    caption_audit.cli.analyze(directory, args)  -> the whole report dict
    caption_audit.report.render_term_report / to_json

`analyze` and the renderers read their settings off an argparse namespace, so
`build_args()` produces a plain object carrying exactly the fields they touch.

The card is a report dict drawn onto a fixed-size PIL canvas. Fixed size is
the point: the node re-renders on every queue, and a canvas that grew with the
number of findings would resize the node in the graph each run. So the
geometry depends on ONE input — `table_rows` — and on nothing in the data.
"""
