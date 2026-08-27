import pytest
from mlbiosig import build_features, samples_labels

# testing if samples exist to build features:


def test_samples_exist():
    with pytest.raises(ValueError, match="No samples were able to be preprocessed."):
        build_features(["test.CSV"], ["biotic"])


# testing biotic-abiotic sample folder structure:


@pytest.mark.parametrize("missing_folder", ["biotic", "abiotic", "missing"])
def text_expected_folders(tmp_path, missing_folder):
    data_dir = tmp_path / "data"
    data_dir.mkdir()  # make dummy folder

    for folder in ["biotic", "abiotic"]:
        if folder != missing_folder:
            (data_dir / folder).mkdir()

    with pytest.raises(FileNotFoundError):
        samples_labels(data_dir=str(data_dir), label_dir=["biotic", "abiotic"])
