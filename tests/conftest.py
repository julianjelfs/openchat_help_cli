from pathlib import Path

import pytest

from ocqa.corpus import load_corpus
from ocqa.evals.golden import load_golden

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def chunks():
    return load_corpus(REPO_ROOT / "corpus")


@pytest.fixture(scope="session")
def golden():
    return load_golden(REPO_ROOT / "evals" / "golden.jsonl")
