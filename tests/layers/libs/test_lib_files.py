"""Golden: Save Image's counter scan, latest_counter (target libs/files.py).

latest_counter(folder, filename, counter_digits, counter_position, output_ext) is 1 + the highest
counter among the files of `folder` that end with `output_ext` and carry `filename`, read at the
end ("last") or the start ("first") of the name. The table pins it over one folder of mixed
names: both positions, no file name, an unknown position (falls back to "last"), other digit
counts, other extensions, non-decimal counters, a missing and an empty folder. Values are
recorded as repr, so an int that turns into another type shows.
"""

import os

import pytest

from _golden import Where, check, check_env

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

GOLDEN = {
    'last': '8',
    'first': '3',
    'first_png': '21',
    'no_filename': '11',
    'no_filename_first': '11',
    'no_filename_3_digits': '1',
    '3_digits': '43',
    'unknown_position': '8',
    'png': '13',
    'extension_case': '101',
    'no_file_with_ext': '1',
    'no_matching_name': '1',
    'other_name': '10',
    'zero_digits': '1',
    'zero_digits_no_filename': '1',
    'empty_folder': '1',
    'missing_folder': '1',
}

WHERE = Where({
    "latest_counter": "libs.files:latest_counter",
})

FILES = [
    "img-0001.webp", "img-0007.webp", "img-00x5.webp", "img-042.webp", "img-0012.png",
    "img-0100.WEBP", "other-0009.webp", "0003.webp", "0010.webp", "0002-img.webp", "0020-img.png",
    "abcd-img.webp", "notes.txt",
]

# name -> (folder: "mixed" | "empty" | "missing", filename, counter_digits, counter_position, output_ext)
CASES = {
    "last": ("mixed", "img", 4, "last", ".webp"),
    "first": ("mixed", "img", 4, "first", ".webp"),
    "first_png": ("mixed", "img", 4, "first", ".png"),
    "no_filename": ("mixed", "", 4, "last", ".webp"),
    "no_filename_first": ("mixed", "", 4, "first", ".webp"),
    "no_filename_3_digits": ("mixed", "", 3, "last", ".webp"),
    "3_digits": ("mixed", "img", 3, "last", ".webp"),
    "unknown_position": ("mixed", "img", 4, "middle", ".webp"),
    "png": ("mixed", "img", 4, "last", ".png"),
    "extension_case": ("mixed", "img", 4, "last", ".WEBP"),
    "no_file_with_ext": ("mixed", "img", 4, "last", ".jpg"),
    "no_matching_name": ("mixed", "nomatch", 4, "last", ".webp"),
    "other_name": ("mixed", "other", 4, "last", ".webp"),
    "zero_digits": ("mixed", "img", 0, "last", ".webp"),
    "zero_digits_no_filename": ("mixed", "", 0, "last", ".webp"),
    "empty_folder": ("empty", "img", 4, "last", ".webp"),
    "missing_folder": ("missing", "img", 4, "last", ".webp"),
}


@pytest.fixture
def folders(tmp_path):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    for name in FILES:
        (mixed / name).write_bytes(b"")
    (tmp_path / "empty").mkdir()
    return {"mixed": str(mixed), "empty": str(tmp_path / "empty"), "missing": str(tmp_path / "missing")}


@pytest.mark.parametrize("name", list(CASES))
def test_latest_counter(name, folders, bcnodes):
    check_env(ENV)
    folder, filename, digits, position, ext = CASES[name]
    assert os.path.isdir(folders[folder]) == (folder != "missing")
    check(GOLDEN, name, repr(WHERE["latest_counter"](folders[folder], filename, digits, position, ext)))
