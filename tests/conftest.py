import numpy as np
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_model():
    """Return a mock SentenceTransformer that returns fixed embeddings."""
    model = MagicMock()
    model.encode.return_value = np.full((1, 384), 0.5, dtype=np.float32)
    return model
