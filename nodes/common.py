"""Shared bits for node definitions. Imports nothing beyond the stdlib."""


class AnyType(str):
    """Wildcard socket type. ComfyUI compares socket types with `!=`, so a
    string that is never "not equal" links to, and validates against, any
    type. The frontend treats the plain "*" as its wildcard."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")


class FlexibleOptionalInputType(dict):
    """An `optional` INPUT_TYPES mapping that claims to hold every key.

    ComfyUI looks an input up with `name in optional` and `optional[name]`;
    answering both for any name lets a node accept inputs that only exist on
    the canvas (slots added by a web extension) under any name, typed as
    `socket_type`. Keys given in `known` are returned as declared."""

    def __init__(self, socket_type, known=None):
        super().__init__()
        self.socket_type = socket_type
        self.known = known or {}
        for k, v in self.known.items():
            self[k] = v

    def __getitem__(self, key):
        if key in self.known:
            return self.known[key]
        return (self.socket_type,)

    def __contains__(self, key):
        return True


def slot_index(pattern, name):
    m = pattern.match(name)
    return int(m.group(1)) if m else float("inf")
