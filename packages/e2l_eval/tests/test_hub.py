from pathlib import Path

import huggingface_hub

from e2l_eval import hub


def test_namespace_prefers_config_then_logged_in_user(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "whoami", lambda: {"name": "someone"})
    assert hub.namespace("team") == "team"
    assert hub.namespace(None) == "someone"


class _Api:
    calls: list = []

    def repo_exists(self, repo_id, repo_type=None):
        _Api.calls.append(("exists", repo_id, repo_type))
        return repo_id.endswith("yes")

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False):
        _Api.calls.append(("create", repo_id, repo_type, private, exist_ok))

    def upload_folder(self, repo_id, folder_path, repo_type=None):
        _Api.calls.append(("upload", repo_id, folder_path, repo_type))


def test_repo_exists_and_push_folder_use_dataset_type(monkeypatch, tmp_path):
    _Api.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    assert hub.repo_exists("a/yes", "dataset") and not hub.repo_exists("a/no")
    hub.push_folder(tmp_path, "a/e2l-x-plan", private=True)
    assert ("exists", "a/yes", "dataset") in _Api.calls
    assert ("exists", "a/no", "model") in _Api.calls
    assert ("create", "a/e2l-x-plan", "dataset", True, True) in _Api.calls
    assert ("upload", "a/e2l-x-plan", str(tmp_path), "dataset") in _Api.calls


def test_download_policy_returns_snapshot_dir(monkeypatch, tmp_path):
    seen = {}

    def fake(repo_id, repo_type=None):
        seen.update(repo_id=repo_id, repo_type=repo_type)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert hub.download_policy("a/b-policy") == Path(tmp_path)
    assert seen == {"repo_id": "a/b-policy", "repo_type": "model"}
