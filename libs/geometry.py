"""Integer size arithmetic for image resizing."""

import math


def round_up_to_multiple(number, multiple):
    if number % multiple == 0:
        return number
    return ((number + multiple - 1) // multiple) * multiple


def short_side_size(height, width, short):
    """(height, width) scaled so the short side is `short`, the long side keeping the aspect: each
    side times short / min(height, width), rounded half to even."""
    k = short / min(height, width)
    return int(round(height * k)), int(round(width * k))


def aspect_ratio(choice, orig_width, orig_height, proportional_width, proportional_height):
    """Width / height of Image Scale By Aspect Ratio's `aspect_ratio` widget: the source's
    ("original"), the proportional widgets' ("custom"), or an "a:b" preset."""
    if choice == "original":
        return orig_width / orig_height
    if choice == "custom":
        return proportional_width / proportional_height
    a, b = choice.split(":")
    return int(a) / int(b)


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
