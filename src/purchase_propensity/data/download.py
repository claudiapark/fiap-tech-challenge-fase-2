"""Download and verify the official UCI dataset."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

CSV_SUFFIX = "online_shoppers_intention.csv"


def fetch_archive(url: str, timeout: int = 60) -> bytes:
    """Fetch a remote archive using an explicit user agent."""
    request = Request(url, headers={"User-Agent": "fiap-purchase-propensity/0.1"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def extract_dataset(archive: bytes) -> bytes:
    """Extract the expected CSV from a ZIP archive."""
    with ZipFile(BytesIO(archive)) as zipped:
        matches = [name for name in zipped.namelist() if name.endswith(CSV_SUFFIX)]
        if len(matches) != 1:
            raise ValueError(f"Expected one dataset CSV, found {len(matches)}")
        return zipped.read(matches[0])


def verify_checksum(content: bytes, expected_sha256: str) -> None:
    """Raise when downloaded content does not match the pinned checksum."""
    actual_sha256 = sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Checksum mismatch: expected {expected_sha256}, got {actual_sha256}")


def download_dataset(url: str, expected_sha256: str, output_path: Path) -> None:
    """Download, validate, and persist the source dataset."""
    dataset = extract_dataset(fetch_archive(url))
    verify_checksum(dataset, expected_sha256)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(dataset)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the dataset download command."""
    args = parse_args()
    download_dataset(args.url, args.sha256, args.output)
    print(f"Dataset verified and saved to {args.output}")


if __name__ == "__main__":
    main()
