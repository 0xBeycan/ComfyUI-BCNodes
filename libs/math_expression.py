"""The whitelisted arithmetic evaluator behind BC_MathExpression.

The expression is parsed with `ast` and only a fixed set of node types and functions is
walked; anything else is a ValueError. The caller resolves what the language cannot know on
its own: size_of(value, name, attr) gives name.width / name.height of an input, and
widget_value(owner, attr) gives Owner.attr, another node's widget.
"""

import ast
import math
import operator
import random

BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Invert: operator.invert,
    ast.Not: lambda a: 0 if a else 1,
}

COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
}

# name -> (min args, max args or None, callable)
FUNCTIONS = {
    "min": (1, None, min),
    "max": (1, None, max),
    "abs": (1, 1, abs),
    "round": (1, 2, lambda a, b=None: round(a, b)),
    "int": (1, 1, int),
    "float": (1, 1, float),
    "pow": (2, 2, pow),
    "sqrt": (1, 1, math.sqrt),
    "floor": (1, 1, math.floor),
    "ceil": (1, 1, math.ceil),
    "randomint": (2, 2, lambda a, b: random.randint(int(a), int(b))),
    "randomchoice": (1, None, lambda *args: random.choice(args)),
    "iif": (3, 3, lambda cond, yes, no: yes if cond else no),
}

# Functions whose result changes between runs with the same arguments.
VOLATILE_FUNCTIONS = ("randomint", "randomchoice")

INPUT_NAMES = ("a", "b", "c")


def _number(value, what):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    raise ValueError(f"{what} is not a number")


class Evaluator:
    def __init__(self, inputs, size_of, widget_value):
        self.inputs = inputs
        self.size_of = size_of
        self.widget_value = widget_value

    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant):
            return _number(node.value, f"constant {node.value!r}")
        if isinstance(node, ast.BinOp):
            op = BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator not allowed: {type(node.op).__name__}")
            return op(self.visit(node.left), self.visit(node.right))
        if isinstance(node, ast.UnaryOp):
            op = UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator not allowed: {type(node.op).__name__}")
            return op(self.visit(node.operand))
        if isinstance(node, ast.BoolOp):
            values = [self.visit(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return 1 if all(values) else 0
            return 1 if any(values) else 0
        if isinstance(node, ast.Compare):
            left = self.visit(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                fn = COMPARE_OPS.get(type(op))
                if fn is None:
                    raise ValueError(f"comparison not allowed: {type(op).__name__}")
                right = self.visit(comparator)
                if not fn(left, right):
                    return 0
                left = right
            return 1
        if isinstance(node, ast.Name):
            if node.id not in INPUT_NAMES:
                raise ValueError(f"unknown name: {node.id}")
            value = self.inputs.get(node.id)
            if value is None:
                raise ValueError(f"input {node.id} is not connected")
            if isinstance(value, (int, float)):
                return _number(value, node.id)
            raise ValueError(f"{node.id} is an IMAGE/LATENT; use {node.id}.width or {node.id}.height")
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name):
                raise ValueError("only name.attribute is allowed")
            owner = node.value.id
            if owner in INPUT_NAMES:
                if node.attr not in ("width", "height"):
                    raise ValueError(f"{owner}.{node.attr}: only width and height are available")
                value = self.inputs.get(owner)
                if value is None:
                    raise ValueError(f"input {owner} is not connected")
                return self.size_of(value, owner, node.attr)
            return _number(self.widget_value(owner, node.attr), f"{owner}.{node.attr}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                name = getattr(node.func, "id", type(node.func).__name__)
                raise ValueError(f"function not allowed: {name}")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
            lo, hi, fn = FUNCTIONS[node.func.id]
            n = len(node.args)
            if n < lo or (hi is not None and n > hi):
                expected = f"{lo}" if hi == lo else (f"{lo} or more" if hi is None else f"{lo} to {hi}")
                raise ValueError(f"{node.func.id}() takes {expected} argument(s), got {n}")
            return fn(*[self.visit(arg) for arg in node.args])
        raise ValueError(f"not allowed in an expression: {type(node).__name__}")


def _parse(expression):
    """The expression's AST, or None when it is blank."""
    text = (expression or "").replace("\r", "").replace("\n", " ").strip()
    if not text:
        return None
    try:
        return ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"cannot parse expression: {e.msg}") from None


def is_volatile(expression):
    """True when the result can differ between runs with the same inputs: a
    random function, or another node's widget (Title.widget), which is not
    part of the Math Expression node's cache signature. A blank or unparsable
    expression is not volatile; evaluate() raises the parse error."""
    try:
        tree = _parse(expression)
    except ValueError:
        return False
    for node in ast.walk(tree) if tree is not None else ():
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in VOLATILE_FUNCTIONS:
            return True
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id not in INPUT_NAMES:
            return True
    return False


def evaluate(expression, inputs, size_of, widget_value):
    tree = _parse(expression)
    if tree is None:
        return 0
    result = Evaluator(inputs or {}, size_of, widget_value).visit(tree)
    if isinstance(result, bool):
        result = int(result)
    if not isinstance(result, (int, float)):
        raise ValueError("expression did not produce a number")
    return result
