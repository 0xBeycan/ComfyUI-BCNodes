"""Golden: Math Expression, entered through the node class only.

For each row of the table: run()'s return value, or the (exception type, message) it raises,
and IS_CHANGED for the same expression. Rows cover every operator (negatives, floats, bitwise,
unary), the non-short-circuit and/or/iif, chained comparisons, rounding, every function, the
IMAGE / MASK / 4-D and 5-D LATENT sizes, widget lookups by title, type and S&R name, a linked
widget, a missing node, bool inputs, constants of every kind, keyword arguments, unknown
functions, subscripts, and an infinite / NaN result reaching int().

The random functions draw from the global `random` module: it is seeded with 0 before every
row, and its state is restored afterwards. No WHERE: nothing is reached around the node. The
messages of Python's own exceptions depend on its version, which ENV pins.
"""

import random

import pytest
import torch

from _golden import check, check_env

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

GOLDEN = {
    'empty': {'run': "{'ui': {'value': [0]}, 'result': (0, 0.0)}", 'IS_CHANGED': "''"},
    'whitespace': {'run': "{'ui': {'value': [0]}, 'result': (0, 0.0)}", 'IS_CHANGED': "' \\n '"},
    'None': {'run': "{'ui': {'value': [0]}, 'result': (0, 0.0)}", 'IS_CHANGED': 'None'},
    'newlines_joined': {'run': "{'ui': {'value': [3]}, 'result': (3, 3.0)}", 'IS_CHANGED': "'\\r\\n1 +\\n2'"},
    'syntax_error': {'run': "('ValueError', 'cannot parse expression: invalid syntax')", 'IS_CHANGED': "'1 +'"},
    'unclosed': {'run': '(\'ValueError\', "cannot parse expression: \'(\' was never closed")', 'IS_CHANGED': "'(1'"},
    'add': {'run': "{'ui': {'value': [3]}, 'result': (3, 3.0)}", 'IS_CHANGED': "'1 + 2'"},
    'sub_negatives': {'run': "{'ui': {'value': [1]}, 'result': (1, 1.0)}", 'IS_CHANGED': "'-3 - -4'"},
    'mul_float': {'run': "{'ui': {'value': [6.0]}, 'result': (6, 6.0)}", 'IS_CHANGED': "'1.5 * 4'"},
    'div': {'run': "{'ui': {'value': [3.5]}, 'result': (3, 3.5)}", 'IS_CHANGED': "'7 / 2'"},
    'div_exact': {'run': "{'ui': {'value': [2.0]}, 'result': (2, 2.0)}", 'IS_CHANGED': "'6 / 3'"},
    'div_zero': {'run': "('ZeroDivisionError', 'division by zero')", 'IS_CHANGED': "'10 / 0'"},
    'floordiv_negative': {'run': "{'ui': {'value': [-4]}, 'result': (-4, -4.0)}", 'IS_CHANGED': "'-7 // 2'"},
    'floordiv_float': {'run': "{'ui': {'value': [3.0]}, 'result': (3, 3.0)}", 'IS_CHANGED': "'7.5 // 2'"},
    'mod_negative': {'run': "{'ui': {'value': [2]}, 'result': (2, 2.0)}", 'IS_CHANGED': "'-7 % 3'"},
    'mod_float': {'run': "{'ui': {'value': [1.5]}, 'result': (1, 1.5)}", 'IS_CHANGED': "'7.5 % 2'"},
    'pow': {'run': "{'ui': {'value': [1024]}, 'result': (1024, 1024.0)}", 'IS_CHANGED': "'2 ** 10'"},
    'pow_negative_exponent': {'run': "{'ui': {'value': [0.5]}, 'result': (0, 0.5)}", 'IS_CHANGED': "'2 ** -1'"},
    'pow_big': {'run': "{'ui': {'value': [1267650600228229401496703205376]}, 'result': (1267650600228229401496703205376, 1.2676506002282294e+30)}", 'IS_CHANGED': "'2 ** 100'"},
    'pow_complex': {'run': "('ValueError', 'expression did not produce a number')", 'IS_CHANGED': "'(-8) ** (1 / 3)'"},
    'bitand': {'run': "{'ui': {'value': [8]}, 'result': (8, 8.0)}", 'IS_CHANGED': "'12 & 10'"},
    'bitor': {'run': "{'ui': {'value': [15]}, 'result': (15, 15.0)}", 'IS_CHANGED': "'12 | 3'"},
    'bitxor': {'run': "{'ui': {'value': [6]}, 'result': (6, 6.0)}", 'IS_CHANGED': "'12 ^ 10'"},
    'lshift': {'run': "{'ui': {'value': [16]}, 'result': (16, 16.0)}", 'IS_CHANGED': "'1 << 4'"},
    'rshift': {'run': "{'ui': {'value': [32]}, 'result': (32, 32.0)}", 'IS_CHANGED': "'256 >> 3'"},
    'bitand_float': {'run': '(\'TypeError\', "unsupported operand type(s) for &: \'float\' and \'int\'")', 'IS_CHANGED': "'1.5 & 1'"},
    'matmul': {'run': "('ValueError', 'operator not allowed: MatMult')", 'IS_CHANGED': "'2 @ 3'"},
    'invert': {'run': "{'ui': {'value': [-6]}, 'result': (-6, -6.0)}", 'IS_CHANGED': "'~5'"},
    'unary_plus': {'run': "{'ui': {'value': [3]}, 'result': (3, 3.0)}", 'IS_CHANGED': "'+3'"},
    'unary_minus_float': {'run': "{'ui': {'value': [-2.5]}, 'result': (-2, -2.5)}", 'IS_CHANGED': "'-2.5'"},
    'not_zero': {'run': "{'ui': {'value': [1]}, 'result': (1, 1.0)}", 'IS_CHANGED': "'not 0'"},
    'not_three': {'run': "{'ui': {'value': [0]}, 'result': (0, 0.0)}", 'IS_CHANGED': "'not 3'"},
    'and': {'run': "{'ui': {'value': [1]}, 'result': (1, 1.0)}", 'IS_CHANGED': "'2 and 3'"},
    'and_zero': {'run': "{'ui': {'value': [0]}, 'result': (0, 0.0)}", 'IS_CHANGED': "'2 and 0'"},
    'or': {'run': "{'ui': {'value': [1]}, 'result': (1, 1.0)}", 'IS_CHANGED': "'0 or 5'"},
    'or_zero': {'run': "{'ui': {'value': [0]}, 'result': (0, 0.0)}", 'IS_CHANGED': "'0 or 0.0'"},
    'and_no_short_circuit': {'run': "('ZeroDivisionError', 'division by zero')", 'IS_CHANGED': "'0 and 1 / 0'"},
    'or_no_short_circuit': {'run': "('ZeroDivisionError', 'division by zero')", 'IS_CHANGED': "'1 or 1 / 0'"},
    'iif': {'run': "{'ui': {'value': [5]}, 'result': (5, 5.0)}", 'IS_CHANGED': "'iif(a > b, a, b)'"},
    'iif_no_short_circuit': {'run': "('ZeroDivisionError', 'division by zero')", 'IS_CHANGED': "'iif(1, 2, 1 / 0)'"},
    'compare_eq_int_float': {'run': "{'ui': {'value': [1]}, 'result': (1, 1.0)}", 'IS_CHANGED': "'2 == 2.0'"},
    'compare_ne': {'run': "{'ui': {'value': [1]}, 'result': (1, 1.0)}", 'IS_CHANGED': "'2 != 3'"},
    'compare_ge_le': {'run': "{'ui': {'value': [1]}, 'result': (1, 1.0)}", 'IS_CHANGED': "'(3 >= 3) + (2 <= 1)'"},
    'chain_true': {'run': "{'ui': {'value': [1]}, 'result': (1, 1.0)}", 'IS_CHANGED': "'1 < 2 < 3'"},
    'chain_false': {'run': "{'ui': {'value': [0]}, 'result': (0, 0.0)}", 'IS_CHANGED': "'1 < 3 < 2'"},
    'chain_stops_at_false': {'run': "{'ui': {'value': [0]}, 'result': (0, 0.0)}", 'IS_CHANGED': "'1 > 2 < 1 / 0'"},
    'compare_is': {'run': "('ValueError', 'comparison not allowed: Is')", 'IS_CHANGED': "'1 is 1'"},
    'compare_in': {'run': "('ValueError', 'comparison not allowed: In')", 'IS_CHANGED': "'1 in 2'"},
    'min': {'run': "{'ui': {'value': [1]}, 'result': (1, 1.0)}", 'IS_CHANGED': "'min(3, 1, 2)'"},
    'min_one_arg': {'run': '(\'TypeError\', "\'int\' object is not iterable")', 'IS_CHANGED': "'min(5)'"},
    'max': {'run': "{'ui': {'value': [2]}, 'result': (2, 2.0)}", 'IS_CHANGED': "'max(1.5, 2)'"},
    'abs': {'run': "{'ui': {'value': [4]}, 'result': (4, 4.0)}", 'IS_CHANGED': "'abs(-4)'"},
    'round_half_even': {'run': "{'ui': {'value': [42]}, 'result': (42, 42.0)}", 'IS_CHANGED': "'round(2.5) + round(3.5) * 10'"},
    'round_digits': {'run': "{'ui': {'value': [2.67]}, 'result': (2, 2.67)}", 'IS_CHANGED': "'round(2.675, 2)'"},
    'round_negative_digits': {'run': "{'ui': {'value': [1200]}, 'result': (1200, 1200.0)}", 'IS_CHANGED': "'round(1234, -2)'"},
    'round_minus_half': {'run': "{'ui': {'value': [0]}, 'result': (0, 0.0)}", 'IS_CHANGED': "'round(-0.5)'"},
    'int': {'run': "{'ui': {'value': [-2]}, 'result': (-2, -2.0)}", 'IS_CHANGED': "'int(-2.7)'"},
    'float': {'run': "{'ui': {'value': [3.0]}, 'result': (3, 3.0)}", 'IS_CHANGED': "'float(3)'"},
    'pow_function': {'run': "{'ui': {'value': [1.4142135623730951]}, 'result': (1, 1.4142135623730951)}", 'IS_CHANGED': "'pow(2, 0.5)'"},
    'sqrt': {'run': "{'ui': {'value': [4.0]}, 'result': (4, 4.0)}", 'IS_CHANGED': "'sqrt(16)'"},
    'sqrt_negative': {'run': "('ValueError', 'math domain error')", 'IS_CHANGED': "'sqrt(-1)'"},
    'floor': {'run': "{'ui': {'value': [-3]}, 'result': (-3, -3.0)}", 'IS_CHANGED': "'floor(-2.5)'"},
    'ceil': {'run': "{'ui': {'value': [3]}, 'result': (3, 3.0)}", 'IS_CHANGED': "'ceil(2.1)'"},
    'randomint': {'run': "{'ui': {'value': [7]}, 'result': (7, 7.0)}", 'IS_CHANGED': 'nan'},
    'randomint_floats': {'run': "{'ui': {'value': [2]}, 'result': (2, 2.0)}", 'IS_CHANGED': 'nan'},
    'randomint_empty_range': {'run': "('ValueError', 'empty range in randrange(5, 2)')", 'IS_CHANGED': 'nan'},
    'randomchoice': {'run': "{'ui': {'value': [2]}, 'result': (2, 2.0)}", 'IS_CHANGED': 'nan'},
    'randomchoice_in_expression': {'run': "{'ui': {'value': [117]}, 'result': (117, 117.0)}", 'IS_CHANGED': 'nan'},
    'too_few_args': {'run': "('ValueError', 'abs() takes 1 argument(s), got 0')", 'IS_CHANGED': "'abs()'"},
    'too_many_args': {'run': "('ValueError', 'round() takes 1 to 2 argument(s), got 3')", 'IS_CHANGED': "'round(1, 2, 3)'"},
    'no_args_min': {'run': "('ValueError', 'min() takes 1 or more argument(s), got 0')", 'IS_CHANGED': "'min()'"},
    'pow_one_arg': {'run': "('ValueError', 'pow() takes 2 argument(s), got 1')", 'IS_CHANGED': "'pow(2)'"},
    'keyword_args': {'run': "('ValueError', 'keyword arguments are not allowed')", 'IS_CHANGED': "'round(2.5, ndigits=1)'"},
    'unknown_function': {'run': "('ValueError', 'function not allowed: exp')", 'IS_CHANGED': "'exp(1)'"},
    'method_call': {'run': "('ValueError', 'function not allowed: Attribute')", 'IS_CHANGED': "'a.width(1)'"},
    'unknown_name': {'run': "('ValueError', 'unknown name: x')", 'IS_CHANGED': "'x + 1'"},
    'not_connected': {'run': "('ValueError', 'input a is not connected')", 'IS_CHANGED': "'a + 1'"},
    'int_inputs': {'run': "{'ui': {'value': [7]}, 'result': (7, 7.0)}", 'IS_CHANGED': "'a + b * c'"},
    'float_input': {'run': "{'ui': {'value': [5.0]}, 'result': (5, 5.0)}", 'IS_CHANGED': "'a * 2'"},
    'bool_inputs': {'run': "{'ui': {'value': [1]}, 'result': (1, 1.0)}", 'IS_CHANGED': "'a + b'"},
    'bool_result': {'run': "{'ui': {'value': [1]}, 'result': (1, 1.0)}", 'IS_CHANGED': "'a'"},
    'string_input': {'run': "('ValueError', 'a is an IMAGE/LATENT; use a.width or a.height')", 'IS_CHANGED': "'a + 1'"},
    'image_as_number': {'run': "('ValueError', 'a is an IMAGE/LATENT; use a.width or a.height')", 'IS_CHANGED': "'a + 1'"},
    'constant_true': {'run': "{'ui': {'value': [2]}, 'result': (2, 2.0)}", 'IS_CHANGED': "'True + True'"},
    'constant_string': {'run': '(\'ValueError\', "constant \'abc\' is not a number")', 'IS_CHANGED': '"\'abc\'"'},
    'constant_none': {'run': "('ValueError', 'constant None is not a number')", 'IS_CHANGED': "'None'"},
    'constant_complex': {'run': "('ValueError', 'constant 1j is not a number')", 'IS_CHANGED': "'1j'"},
    'subscript': {'run': "('ValueError', 'not allowed in an expression: Subscript')", 'IS_CHANGED': "'a[0]'"},
    'list': {'run': "('ValueError', 'not allowed in an expression: List')", 'IS_CHANGED': "'[1, 2]'"},
    'if_expression': {'run': "('ValueError', 'not allowed in an expression: IfExp')", 'IS_CHANGED': "'1 if a else 2'"},
    'nested_attribute': {'run': "('ValueError', 'only name.attribute is allowed')", 'IS_CHANGED': "'a.b.c'"},
    'image_size': {'run': "{'ui': {'value': [96048]}, 'result': (96048, 96048.0)}", 'IS_CHANGED': "'a.width * 1000 + a.height'"},
    'mask_size': {'run': "{'ui': {'value': [72040]}, 'result': (72040, 72040.0)}", 'IS_CHANGED': "'a.width * 1000 + a.height'"},
    'mask_2d': {'run': "('ValueError', 'a.width: a is not an IMAGE or LATENT')", 'IS_CHANGED': "'a.width'"},
    'latent_4d': {'run': "{'ui': {'value': [96048]}, 'result': (96048, 96048.0)}", 'IS_CHANGED': "'b.width * 1000 + b.height'"},
    'latent_5d': {'run': "{'ui': {'value': [48024]}, 'result': (48024, 48024.0)}", 'IS_CHANGED': "'c.width * 1000 + c.height'"},
    'size_of_number': {'run': "('ValueError', 'a.height: a is not an IMAGE or LATENT')", 'IS_CHANGED': "'a.height'"},
    'size_not_connected': {'run': "('ValueError', 'input a is not connected')", 'IS_CHANGED': "'a.width'"},
    'size_other_attribute': {'run': "('ValueError', 'a.depth: only width and height are available')", 'IS_CHANGED': "'a.depth'"},
    'widget_by_type_first_match': {'run': "{'ui': {'value': [20]}, 'result': (20, 20.0)}", 'IS_CHANGED': 'nan'},
    'widget_by_title': {'run': "{'ui': {'value': [37.5]}, 'result': (37, 37.5)}", 'IS_CHANGED': 'nan'},
    'widget_by_s_and_r': {'run': "{'ui': {'value': [0.6842105263157895]}, 'result': (0, 0.6842105263157895)}", 'IS_CHANGED': 'nan'},
    'widget_bool': {'run': "{'ui': {'value': [1]}, 'result': (1, 1.0)}", 'IS_CHANGED': 'nan'},
    'widget_negative_zero': {'run': "{'ui': {'value': [-0.0]}, 'result': (0, -0.0)}", 'IS_CHANGED': 'nan'},
    'widget_linked': {'run': "('ValueError', 'KSampler.model is linked, not a widget; wire it into a, b or c instead')", 'IS_CHANGED': 'nan'},
    'widget_missing': {'run': "('ValueError', 'widget not found: KSampler.denoise')", 'IS_CHANGED': 'nan'},
    'widget_string': {'run': "('ValueError', 'Loader.ckpt_name is not a number')", 'IS_CHANGED': 'nan'},
    'node_missing': {'run': "('ValueError', 'node not found: Nope')", 'IS_CHANGED': 'nan'},
    'no_workflow': {'run': "('ValueError', 'node not found: KSampler')", 'IS_CHANGED': 'nan'},
    'no_prompt': {'run': "('ValueError', 'widget not found: KSampler.steps')", 'IS_CHANGED': 'nan'},
    'widget_and_input': {'run': "{'ui': {'value': [2]}, 'result': (2, 2.0)}", 'IS_CHANGED': 'nan'},
    'inf_constant': {'run': "('OverflowError', 'cannot convert float infinity to integer')", 'IS_CHANGED': "'1e400'"},
    'inf_input': {'run': "('OverflowError', 'cannot convert float infinity to integer')", 'IS_CHANGED': "'a * 1'"},
    'nan_input': {'run': "('ValueError', 'cannot convert float NaN to integer')", 'IS_CHANGED': "'a'"},
    'negative_zero': {'run': "{'ui': {'value': [-0.0]}, 'result': (0, -0.0)}", 'IS_CHANGED': "'-0.0'"},
    'random_sequence': [(864, 864.0), (394, 394.0), (776, 776.0), (911, 911.0), (430, 430.0)],
}

WORKFLOW = {"workflow": {"nodes": [
    {"id": 7, "type": "KSampler", "title": "Main", "properties": {"Node name for S&R": "KSampler"}},
    {"id": 8, "type": "EmptyLatentImage", "properties": {"Node name for S&R": "Latent_SR"}},
    {"id": 9, "type": "CheckpointLoaderSimple", "title": "Loader"},
    {"id": 10, "type": "KSampler", "title": "Second"},
    {"id": 11, "type": "Switch", "properties": None},
]}}
PROMPT = {
    "7": {"inputs": {"steps": 20, "cfg": 7.5, "model": ["9", 0], "add_noise": True}},
    "8": {"inputs": {"width": 832, "height": 1216, "batch_size": 1}},
    "9": {"inputs": {"ckpt_name": "model_a.safetensors"}},
    "10": {"inputs": {"steps": 30}},
    "11": {"inputs": {"value": -0.0}},
}
LOOKUP = {"prompt": PROMPT, "extra_pnginfo": WORKFLOW}

IMAGE = torch.zeros((2, 48, 96, 3))
MASK = torch.zeros((1, 40, 72))
MASK_2D = torch.zeros((40, 72))
LATENT_4D = {"samples": torch.zeros((1, 4, 6, 12))}
LATENT_5D = {"samples": torch.zeros((1, 16, 3, 6, 12))}

# name -> (expression, keyword inputs of run())
CASES = {
    # blank and parsing
    "empty": ("", {}),
    "whitespace": (" \n ", {}),
    "None": (None, {}),
    "newlines_joined": ("\r\n1 +\n2", {}),
    "syntax_error": ("1 +", {}),
    "unclosed": ("(1", {}),
    # binary operators
    "add": ("1 + 2", {}),
    "sub_negatives": ("-3 - -4", {}),
    "mul_float": ("1.5 * 4", {}),
    "div": ("7 / 2", {}),
    "div_exact": ("6 / 3", {}),
    "div_zero": ("10 / 0", {}),
    "floordiv_negative": ("-7 // 2", {}),
    "floordiv_float": ("7.5 // 2", {}),
    "mod_negative": ("-7 % 3", {}),
    "mod_float": ("7.5 % 2", {}),
    "pow": ("2 ** 10", {}),
    "pow_negative_exponent": ("2 ** -1", {}),
    "pow_big": ("2 ** 100", {}),
    "pow_complex": ("(-8) ** (1 / 3)", {}),
    "bitand": ("12 & 10", {}),
    "bitor": ("12 | 3", {}),
    "bitxor": ("12 ^ 10", {}),
    "lshift": ("1 << 4", {}),
    "rshift": ("256 >> 3", {}),
    "bitand_float": ("1.5 & 1", {}),
    "matmul": ("2 @ 3", {}),
    # unary operators
    "invert": ("~5", {}),
    "unary_plus": ("+3", {}),
    "unary_minus_float": ("-2.5", {}),
    "not_zero": ("not 0", {}),
    "not_three": ("not 3", {}),
    # boolean operators: every operand is evaluated
    "and": ("2 and 3", {}),
    "and_zero": ("2 and 0", {}),
    "or": ("0 or 5", {}),
    "or_zero": ("0 or 0.0", {}),
    "and_no_short_circuit": ("0 and 1 / 0", {}),
    "or_no_short_circuit": ("1 or 1 / 0", {}),
    "iif": ("iif(a > b, a, b)", {"a": 3, "b": 5}),
    "iif_no_short_circuit": ("iif(1, 2, 1 / 0)", {}),
    # comparisons
    "compare_eq_int_float": ("2 == 2.0", {}),
    "compare_ne": ("2 != 3", {}),
    "compare_ge_le": ("(3 >= 3) + (2 <= 1)", {}),
    "chain_true": ("1 < 2 < 3", {}),
    "chain_false": ("1 < 3 < 2", {}),
    "chain_stops_at_false": ("1 > 2 < 1 / 0", {}),
    "compare_is": ("1 is 1", {}),
    "compare_in": ("1 in 2", {}),
    # functions
    "min": ("min(3, 1, 2)", {}),
    "min_one_arg": ("min(5)", {}),
    "max": ("max(1.5, 2)", {}),
    "abs": ("abs(-4)", {}),
    "round_half_even": ("round(2.5) + round(3.5) * 10", {}),
    "round_digits": ("round(2.675, 2)", {}),
    "round_negative_digits": ("round(1234, -2)", {}),
    "round_minus_half": ("round(-0.5)", {}),
    "int": ("int(-2.7)", {}),
    "float": ("float(3)", {}),
    "pow_function": ("pow(2, 0.5)", {}),
    "sqrt": ("sqrt(16)", {}),
    "sqrt_negative": ("sqrt(-1)", {}),
    "floor": ("floor(-2.5)", {}),
    "ceil": ("ceil(2.1)", {}),
    "randomint": ("randomint(1, 10)", {}),
    "randomint_floats": ("randomint(1.9, 3.9)", {}),
    "randomint_empty_range": ("randomint(5, 1)", {}),
    "randomchoice": ("randomchoice(1, 2, 3)", {}),
    "randomchoice_in_expression": ("randomchoice(a, b) + randomint(0, 100)", {"a": 10, "b": 20}),
    "too_few_args": ("abs()", {}),
    "too_many_args": ("round(1, 2, 3)", {}),
    "no_args_min": ("min()", {}),
    "pow_one_arg": ("pow(2)", {}),
    "keyword_args": ("round(2.5, ndigits=1)", {}),
    "unknown_function": ("exp(1)", {}),
    "method_call": ("a.width(1)", {"a": IMAGE}),
    # names, constants, structure
    "unknown_name": ("x + 1", {}),
    "not_connected": ("a + 1", {}),
    "int_inputs": ("a + b * c", {"a": 1, "b": 2, "c": 3}),
    "float_input": ("a * 2", {"a": 2.5}),
    "bool_inputs": ("a + b", {"a": True, "b": False}),
    "bool_result": ("a", {"a": True}),
    "string_input": ("a + 1", {"a": "5"}),
    "image_as_number": ("a + 1", {"a": IMAGE}),
    "constant_true": ("True + True", {}),
    "constant_string": ("'abc'", {}),
    "constant_none": ("None", {}),
    "constant_complex": ("1j", {}),
    "subscript": ("a[0]", {"a": 1}),
    "list": ("[1, 2]", {}),
    "if_expression": ("1 if a else 2", {"a": 1}),
    "nested_attribute": ("a.b.c", {}),
    # sizes
    "image_size": ("a.width * 1000 + a.height", {"a": IMAGE}),
    "mask_size": ("a.width * 1000 + a.height", {"a": MASK}),
    "mask_2d": ("a.width", {"a": MASK_2D}),
    "latent_4d": ("b.width * 1000 + b.height", {"b": LATENT_4D}),
    "latent_5d": ("c.width * 1000 + c.height", {"c": LATENT_5D}),
    "size_of_number": ("a.height", {"a": 5}),
    "size_not_connected": ("a.width", {}),
    "size_other_attribute": ("a.depth", {"a": IMAGE}),
    # widgets of other nodes
    "widget_by_type_first_match": ("KSampler.steps", LOOKUP),
    "widget_by_title": ("Main.cfg + Second.steps", LOOKUP),
    "widget_by_s_and_r": ("Latent_SR.width / Latent_SR.height", LOOKUP),
    "widget_bool": ("KSampler.add_noise", LOOKUP),
    "widget_negative_zero": ("Switch.value", LOOKUP),
    "widget_linked": ("KSampler.model", LOOKUP),
    "widget_missing": ("KSampler.denoise", LOOKUP),
    "widget_string": ("Loader.ckpt_name", LOOKUP),
    "node_missing": ("Nope.width", LOOKUP),
    "no_workflow": ("KSampler.steps", {"prompt": PROMPT}),
    "no_prompt": ("KSampler.steps", {"extra_pnginfo": WORKFLOW}),
    "widget_and_input": ("a + EmptyLatentImage.batch_size", {"a": 1, **LOOKUP}),
    # int() of the result
    "inf_constant": ("1e400", {}),
    "inf_input": ("a * 1", {"a": float("inf")}),
    "nan_input": ("a", {"a": float("nan")}),
    "negative_zero": ("-0.0", {}),
}


@pytest.fixture
def seeded_random():
    state = random.getstate()
    random.seed(0)
    yield
    random.setstate(state)


def _outcome(fn):
    try:
        return repr(fn())
    except Exception as e:
        return repr((type(e).__name__, str(e)))


@pytest.mark.parametrize("name", list(CASES))
def test_run(name, bcnodes, seeded_random):
    check_env(ENV)
    expression, kwargs = CASES[name]
    node = bcnodes["math_expression"].MathExpression
    check(GOLDEN, name, {
        "run": _outcome(lambda: node().run(expression, **kwargs)),
        "IS_CHANGED": _outcome(lambda: node.IS_CHANGED(expression=expression, **{k: v for k, v in kwargs.items() if k in ("a", "b", "c")})),
    })


def test_run_repeats_draws_from_the_global_rng(bcnodes, seeded_random):
    node = bcnodes["math_expression"].MathExpression()
    check(GOLDEN, "random_sequence", [node.run("randomint(0, 1000)")["result"] for _ in range(5)])
