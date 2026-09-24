"""Model registry: a family key maps model names to their entries, in registration order.

A model package registers its members when it is imported; models/__init__.py imports every
such package, so any import of this module has filled the registry before a lookup. Node combo
lists come from names(), name lookups from get().
"""

MATTING = "matting"

_FAMILIES = {}


def register(family: str, name: str, model: object) -> None:
    members = _FAMILIES.setdefault(family, {})
    if name in members:
        raise ValueError(f"{family} model {name!r} is already registered")
    members[name] = model


def names(family: str) -> list[str]:
    """A new list of the family's names, in registration order; KeyError for an unknown family."""
    return list(_FAMILIES[family])


def get(family: str, name: str) -> object:
    """The entry registered under name; KeyError(name) for an unknown name."""
    return _FAMILIES[family][name]
