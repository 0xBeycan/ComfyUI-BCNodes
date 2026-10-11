"""Weight download with console progress, shared by the model packages. The transport
(resume, hosts, tokens) is libs/download.py."""

import os

from ...libs.download import download_file


def fetch_with_progress(url, path, label):
    """Download url to path, printing the start line and every 10% under `label`; returns path."""
    state = {"step": -1}

    def on_progress(downloaded, total):
        step = int(10 * downloaded / total) if total else 0
        if step != state["step"]:
            state["step"] = step
            print(f"[BCNodes] {label} {step * 10}% ({downloaded / 1e6:.0f}/{total / 1e6:.0f} MB)")

    print(f"[BCNodes] downloading {url} -> {path}")
    download_file(url, path, on_progress)
    return path


def model_file(folder_key, filename, url):
    """The file `filename` in ComfyUI's `folder_key` folders; when none holds it, downloaded from `url` into the
    first of them (ComfyUI's default, or the one an extra_model_paths entry marks is_default). A key ComfyUI does
    not register is added as models/<folder_key> (background_removal and geometry_estimation on builds older than
    the ones that register them); for a key it has, add_model_folder_path is a no-op."""
    import folder_paths

    folder_paths.add_model_folder_path(folder_key, os.path.join(folder_paths.models_dir, folder_key))
    found = folder_paths.get_full_path(folder_key, filename)
    if found is not None:
        return found
    return fetch_with_progress(url, os.path.join(folder_paths.get_folder_paths(folder_key)[0], filename), filename)
