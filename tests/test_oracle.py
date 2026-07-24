import hashlib
import json
from pathlib import Path

ASSETS = Path(__file__).parent / "mgcv_reference" / "assets"


def test_manifest_pins_the_mgcv_oracle() -> None:
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    assert manifest["mgcv_version"] == "1.9.4"
    assert manifest["mgcv_commit"] == "1b6a4c8374612da27e36420b4459e93acb183f2d"
    for case, metadata in manifest["cases"].items():
        digest = hashlib.sha256((ASSETS / f"{case}.npz").read_bytes()).hexdigest()
        assert digest == metadata["sha256"]
