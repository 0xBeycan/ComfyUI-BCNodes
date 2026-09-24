"""File helpers: the next image counter, continued from the files already in a folder."""

import os

COUNTER_POSITIONS = ["last", "first"]


def latest_counter(folder, filename, counter_digits, counter_position, output_ext):
    """1 + the highest counter among the files in `folder` that carry
    `filename` and `output_ext`; 1 when there are none. Gaps are ignored."""
    if not os.path.isdir(folder):
        return 1
    files = [f for f in os.listdir(folder) if f.endswith(output_ext)]
    if not files:
        return 1
    if counter_position not in COUNTER_POSITIONS:
        counter_position = COUNTER_POSITIONS[0]
    ext_len = len(output_ext)
    if not filename:
        counters = [int(f[:counter_digits]) if f[:counter_digits].isdecimal() and len(f) == counter_digits + ext_len else 0
                    for f in files]
    elif counter_position == "last":
        counters = [int(f[-(ext_len + counter_digits):-ext_len]) if f[-(ext_len + counter_digits):-ext_len].isdecimal() else 0
                    for f in files if f.startswith(filename)]
    else:
        counters = [int(f[:counter_digits]) if f[:counter_digits].isdecimal() else 0
                    for f in files if f[counter_digits + 1:].startswith(filename)]
    return max(counters) + 1 if counters else 1
