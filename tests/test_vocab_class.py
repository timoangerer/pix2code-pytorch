import pytest
from pathlib import Path
from vocab import Vocab


@pytest.fixture
def valid_vocab_path():
    return Path(Path(__file__).parent, "vocab.txt")


def test_vocab_creation(valid_vocab_path):
    vocab = Vocab(valid_vocab_path)
    assert vocab.get_id_by_token(vocab.get_padding_token())
