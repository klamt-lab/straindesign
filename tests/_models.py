"""Model loading for the test suite: GitHub-hosted copy first, BiGG last.

`cobra.io.load_model` downloads from BiGG on every fresh machine (CI runners
have an empty cache), and that download is fragile: cobra 0.30 requests
``http://bigg.ucsd.edu`` without following the 301 to https, and BiGG has
since moved to https://bigg.bio. The models the suite needs are therefore
kept on this repository's ``host_gifs`` asset branch and downloaded from
there into ``tests/models`` (git-ignored), which serves as the cache for
later runs. BiGG itself is only tried for models missing from the branch.
"""

import gzip
import io
import logging
from pathlib import Path

from cobra.io import read_sbml_model

MODEL_DIR = Path(__file__).resolve().parent / "models"
ASSET_BRANCH = "https://raw.githubusercontent.com/klamt-lab/straindesign/host_gifs/models/{}.xml.gz"
BIGG_MIRROR = "https://bigg.ucsd.edu/static/models/{}.xml.gz"

logger = logging.getLogger(__name__)


def _model_from_gzip(data: bytes):
    with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as handle:
        return read_sbml_model(io.StringIO(handle.read()))


def _download(url: str) -> bytes:
    import httpx
    response = httpx.get(url, follow_redirects=True, timeout=120)
    response.raise_for_status()
    return response.content


def load_test_model(model_id: str):
    """Return the model `model_id` (a BiGG identifier such as ``e_coli_core``).

    Order: ``tests/models/<id>.xml.gz`` if present, then the ``host_gifs``
    branch (cached into that directory), then cobra's ``load_model`` (its
    bundled models such as ``textbook``, then BiGG and BioModels through its
    own cache), then a redirect-following download from BiGG.
    """
    local = MODEL_DIR / f"{model_id}.xml.gz"
    if local.is_file():
        return _model_from_gzip(local.read_bytes())
    errors = []
    try:
        data = _download(ASSET_BRANCH.format(model_id))
        MODEL_DIR.mkdir(exist_ok=True)
        local.write_bytes(data)
        return _model_from_gzip(data)
    except Exception as err:
        errors.append(f"{ASSET_BRANCH.format(model_id)}: {err}")
    try:
        from cobra.io import load_model
        return load_model(model_id)
    except Exception as err:  # cobra wraps every network failure in RuntimeError
        errors.append(f"cobra.io.load_model: {err}")
    try:
        return _model_from_gzip(_download(BIGG_MIRROR.format(model_id)))
    except Exception as err:
        errors.append(f"{BIGG_MIRROR.format(model_id)}: {err}")
    raise RuntimeError(f"Model '{model_id}' could not be obtained:\n  " + "\n  ".join(errors))
