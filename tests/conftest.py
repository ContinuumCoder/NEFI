import os

import pytest
import torch

os.environ.setdefault("NEFI_DEVICE", "cpu")
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield
