"""Tests for deterministic dataset acquisition."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from purchase_propensity.data import download
from purchase_propensity.data.download import download_dataset, extract_dataset, verify_checksum


def build_archive(content: bytes) -> bytes:
    """Create an in-memory archive matching the UCI filename."""
    buffer = BytesIO()
    with ZipFile(buffer, "w") as zipped:
        zipped.writestr("online_shoppers_intention.csv", content)
    return buffer.getvalue()


def test_extract_and_verify_dataset() -> None:
    content = b"Revenue\nTRUE\n"
    extracted = extract_dataset(build_archive(content))
    verify_checksum(extracted, sha256(content).hexdigest())
    assert extracted == content


def test_checksum_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksum(b"dataset", "0" * 64)


def test_download_dataset_writes_verified_content(tmp_path: Path, monkeypatch) -> None:
    content = b"Revenue\nFALSE\n"
    monkeypatch.setattr(download, "fetch_archive", lambda _: build_archive(content))
    output = tmp_path / "data" / "dataset.csv"
    download_dataset("https://example.test/data.zip", sha256(content).hexdigest(), output)
    assert output.read_bytes() == content


def test_download_main_delegates_to_service(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "dataset.csv"
    args = SimpleNamespace(url="url", sha256="hash", output=output)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(download, "parse_args", lambda: args)
    monkeypatch.setattr(download, "download_dataset", lambda *values: calls.append(values))
    download.main()
    assert calls == [("url", "hash", output)]
    assert str(output) in capsys.readouterr().out
