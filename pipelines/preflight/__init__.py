"""PreFlight: how Instagram, TikTok and X would likely treat an image or a video before it is published
(removal, or reach suppression), one module per concern:

    prompts    the observation prompt (verbatim, versioned) and its schema: validation and coercion
    observe    PreFlight Observe's flow: frames sampled from the batch, generate -> parse -> one retry ->
               validate -> meta; fail closed
    rules      the rules engine: observation + caption -> per-platform verdict ranges, the summary text
    feedback   the append-only feedback store: predictions and outcomes, joined at read time
    report     PreFlight Report's flow: observation JSON -> report, summary, logged record id
    outcome    PreFlight Outcome's flow and its record list
    calibrate  PreFlight Calibrate's two tables over the store

The model is a pure observation sensor; only rules judges. Nothing here imports ComfyUI: the node layer
(nodes/preflight.py) hands in the generate callable and the store's path.
"""
