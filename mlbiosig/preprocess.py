# Imports:
import pandas as pd
import os
from pathlib import Path
from tqdm.auto import tqdm

# Required column names in raw py-GC-MS CSV exported files:
expected_cols = ["point", "masstocharge", "counts"]

# Expected folder structure for obtaining sample CSV files:
# Filepaths are hardcoded, and assume specific repo structure
root = Path(__file__).parent.parent  # file -> mlbiosig -> root
data_folder = str(root / "data")  # agnostic to where \data is
label_folders = ["biotic", "abiotic"]


def load_clean_sample(filepath):
    """
    Loading and cleaning single py-GC-MS CSV file.

    Checks and corrects erroneous py-GC-MS CSV headers with
    expected column names, and cleans non-physical negative
    intensities by clipping at zero.

    Parameters
    ----------
    filepath : str
        Path to raw py-GC-MS CSV file exported from Agilent
        MassHunter Qualitative Analysis Navigator.

    Returns
    -------
    pandas.DataFrame
        Cleaned mass spectral data with columns for point index,
        mass-to-charge ratio (m/z), and intensity counts.
    """
    df = pd.read_csv(
        filepath,  # file path
        comment="#",  # ignore lines beginning with '#'
        names=expected_cols,
    )  # replace header with required columns

    # Clipping non-physical negative intensities:
    df["counts"] = df["counts"].clip(lower=0)
    df = df[df["counts"] > 0]

    return df


def py_ms_vector(df, sample):
    """
    Creates py-MS vector from cleaned py-GC-MS mass spectral
    data.

    Sums intensities across all scans per m/z per py-GC-MS CSV file,
    collapsing GC component and generating py-MS vector.
    Max-normalising summed intensities per m/z for feature creation.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned mass spectral data.
    sample : str
        py-GC-MS sample name for labelling created py-MS vector.

    Returns
    -------
    pandas.DataFrame
        Summed and normalised intensities per m/z.
    pandas.Series
        Normalised py-MS vector indexed by m/z and named
        after parent py-GC-MS sample file.
    """
    # Sum intensities per m/z:
    sum_spectrum = df.groupby("masstocharge")["counts"].sum().reset_index()
    sum_spectrum.columns = ["m_z", "intensity"]

    # Normalise summed intensities per m/z:
    sum_spectrum["intensity_norm"] = (
        sum_spectrum["intensity"] / sum_spectrum["intensity"].max()
    )

    # Create py-MS vector:
    vector = sum_spectrum.set_index("m_z")["intensity_norm"]
    vector.name = sample  # name vector after its parent py-GC-MS CSV filename

    return sum_spectrum, vector


def artefact_checker(sum_spectrum, sample, flags):
    """
    Flag py-MS vector parent samples with unusual spectral
    profiles.

    Compile names of unusual samples in `flags` if there are
    too few distinct m/z values across sample.

    Parameters
    ----------
    sum_spectrum : pandas.DataFrame
        Summed intensities per m/z for a single sample.
    sample : str
        py-GC-MS sample name.
    flags : list
        List of (sample name, reason) tuples to be appended
        to.

    Returns
    -------
    list
        Updated flags list.
    """
    sorted_intensities = sum_spectrum["intensity"].sort_values(ascending=False).values

    # Checking if too few m/z values:
    if len(sorted_intensities) < 5:
        flags.append((sample, ": Less than 5 m/z peaks detected across sample."))

        return flags


def samples_labels(data_dir=data_folder, label_dir=label_folders):
    """
    Reads biotic and abiotic subfolders within main data directory
    to build list of py-GC-MS sample CSV filepaths and their
    corresponding labels.

    Parameters
    ----------
    data_dir : str
        Path to parent data folder containing biotic and abiotic
        subfolders.
    label_dir : list[str]
        Names of biotic and abiotic subfolders used for corresponding
        class labels for sample files.

    Returns
    -------
    list[str]
        Filepaths to both biotic and abiotic CSV files found across
        both subfolders, respectively.
    list[str]
        Corresponding biotic/abiotic label for each filepath, taken from
        the subfolder name.
    """
    filepaths = []
    labels = []

    for label in label_dir:
        # Build folder path i.e., data/biotic:
        folder_path = os.path.join(data_dir, label)

        # Check for expected folder structure:
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(rf"Expected folder not found: {folder_path}")

        all_files = os.listdir(folder_path)  # lists folder items

        for file in all_files:
            if file.endswith(".CSV"):
                file_path = folder_path + "/" + file
                filepaths.append(file_path)
                labels.append(label)

    return filepaths, labels


def build_features(filepaths, labels):
    """
    Main function for iterating over all sample files, computing and
    collecting normalised py-MS vectors as individual features, and
    stacking into a single feature matrix.

    Parameters
    ----------
    filepaths : list[str]
        Paths to all raw py-GC-MS CSV files to be preprocessed.
    labels : list[str]
        Corresponding biotic/abiotic labels for each filepath.

    Returns
    -------
    pandas.DataFrame
        Stacked py-MS vectors, such that there is 1 row per sample,
        1 column per m/z value with additional label column.
    list
        List of (sample name, reason) tuples for sample that are
        flagged or failed to be preprocessed.
    """
    vectors = []
    flags = []

    for filepath, label in tqdm(
        zip(filepaths, labels),
        total=len(filepaths),
        desc="Preprocessing py-GC-MS samples",
        unit="sample",
    ):
        # strip py-GC-MS CSV filenames into sample labels:
        sample = filepath.split("/")[-1].replace(".CSV", "")

        try:
            df = load_clean_sample(filepath)  # load py-GC-MS spectra
            sum_spectrum, vector = py_ms_vector(df, sample)  # generate vectors
            artefact_checker(sum_spectrum, sample, flags)  # flag samples

            # sets column names = m/z, rows = corresponding vector values
            # for each sample:
            vector_df = pd.DataFrame(
                [vector.values], columns=vector.index, index=[sample]
            )
            vector_df["label"] = label  # adds label column for biotic/abiotic
            vectors.append(vector_df)  # appends single row per sample
        except Exception as reason:
            flags.append((sample, f"Failed to process: {reason}"))
            continue

    if not vectors:
        raise ValueError("No samples were able to be preprocessed.")

    # Stack rows per sample into pd.DataFrame, filling m/z columns for samples
    # with no corresponding vector values with zero:
    feature_matrix = pd.concat(vectors).fillna(0)

    return feature_matrix, flags
