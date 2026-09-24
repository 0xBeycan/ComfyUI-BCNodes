"""Weight download with console progress, shared by the model packages. The transport
(resume, hosts, tokens) is libs/download.py."""

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
