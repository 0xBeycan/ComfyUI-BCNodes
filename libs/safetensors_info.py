"""What a weights file holds, from its header only: no tensor is read, no model is built.

A .safetensors file starts with an 8-byte little-endian header length and a JSON header that lists
every tensor's dtype, shape and byte range. Other formats (.ckpt, .pt, .gguf, ...) give their file
size only, and say so.
"""

import json
import math
import os
import struct

# Header lengths above this are not a safetensors header (the format caps it at 100 MB).
_MAX_HEADER = 100 * 1024 * 1024


def weights_info(path):
    """{"bytes", "params", "dtypes": {dtype: bytes}, "source"}: "header" for a safetensors file,
    "file size" otherwise. Raises OSError when the file cannot be read and ValueError on a
    malformed header."""
    if not path.lower().endswith((".safetensors", ".sft")):
        return {"bytes": os.path.getsize(path), "params": None, "dtypes": {}, "source": "file size"}
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        if length > _MAX_HEADER:
            raise ValueError(f"{path}: header length {length} is not a safetensors header")
        header = json.loads(f.read(length))
    total, params, dtypes = 0, 0, {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        total += end - start
        params += math.prod(entry["shape"])
        dtypes[entry["dtype"]] = dtypes.get(entry["dtype"], 0) + end - start
    return {"bytes": total, "params": params, "dtypes": dtypes, "source": "header"}
