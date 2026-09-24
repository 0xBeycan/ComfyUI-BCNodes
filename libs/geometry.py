"""Integer size arithmetic for image resizing."""

import math


def round_up_to_multiple(number, multiple):
    if number % multiple == 0:
        return number
    return ((number + multiple - 1) // multiple) * multiple


def target_size(orig_width, orig_height, ratio, scale_to_side, scale_to_length):
    """Output (width, height) before rounding. Every branch truncates with
    int(), and the expressions are kept in the original's operand order so the
    float results (and therefore the truncation) match it exactly."""
    if ratio > 1:
        if scale_to_side == "longest":
            width = scale_to_length
            height = int(width / ratio)
        elif scale_to_side == "shortest":
            height = scale_to_length
            width = int(height * ratio)
        elif scale_to_side == "width":
            width = scale_to_length
            height = int(width / ratio)
        elif scale_to_side == "height":
            height = scale_to_length
            width = int(height * ratio)
        elif scale_to_side == "total_pixel(kilo pixel)":
            width = math.sqrt(ratio * scale_to_length * 1000)
            height = width / ratio
            width, height = int(width), int(height)
        else:
            width = orig_width
            height = int(width / ratio)
    else:
        if scale_to_side == "longest":
            height = scale_to_length
            width = int(height * ratio)
        elif scale_to_side == "shortest":
            width = scale_to_length
            height = int(width / ratio)
        elif scale_to_side == "width":
            width = scale_to_length
            height = int(width / ratio)
        elif scale_to_side == "height":
            height = scale_to_length
            width = int(height * ratio)
        elif scale_to_side == "total_pixel(kilo pixel)":
            width = math.sqrt(ratio * scale_to_length * 1000)
            height = width / ratio
            width, height = int(width), int(height)
        else:
            height = orig_height
            width = int(height * ratio)
    return width, height
