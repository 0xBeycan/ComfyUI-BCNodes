"""libs/download.py's token store with the Claude node's Anthropic key: saved and read back (stripped), reported only
as present, removed by an empty value, the file readable by its owner only; no download URL, Anthropic's own host
included, ever gets it (it has no HOSTS entry), while the Hugging Face token still goes to huggingface.co.
"""

import os
import stat
import sys

import pytest


@pytest.fixture
def dl(bcnodes, comfy_stubs, monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules["folder_paths"], "get_user_directory", lambda: str(tmp_path / "user"))
    return bcnodes["libs.download"]


def test_anthropic_key_kept_reported_as_present_and_removed(dl):
    assert dl.tokens_present() == {"huggingface": False, "civitai": False, "anthropic": False}
    dl.save_tokens({"anthropic": " sk-ant-test \n", "huggingface": "hf_x"})
    assert dl.read_tokens() == {"anthropic": "sk-ant-test", "huggingface": "hf_x"}
    assert dl.tokens_present() == {"huggingface": True, "civitai": False, "anthropic": True}
    assert stat.S_IMODE(os.stat(dl.tokens_file()).st_mode) == 0o600
    dl.save_tokens({"anthropic": ""})
    assert dl.read_tokens() == {"huggingface": "hf_x"}


def test_no_download_gets_the_anthropic_key(dl):
    dl.save_tokens({"anthropic": "sk-ant-test", "huggingface": "hf_x"})
    for url in ("https://api.anthropic.com/v1/files/x", "https://example.com/model.safetensors"):
        request_url, headers = dl._auth(url)
        assert request_url == url and "Authorization" not in headers
    assert dl._auth("https://huggingface.co/a/b/resolve/main/c.safetensors")[1]["Authorization"] == "Bearer hf_x"
