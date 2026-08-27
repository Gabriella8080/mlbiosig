import os


def demo_samples(sample_path, data_dir="../../data/"):
    """
    Reads from designated text file specifying which biotic and abiotic
    samples to use, and returns their filepaths and labels. This structure
    allows for reproducible selection of sample subsets without hardcoding
    sample names in the `mlbiosig` demonstration notebook.

    Parameters
    ----------
    sample_path : str
        Path to the demonstration sample text file.
    data_dir : str
        Path to parent data folder containing biotic and abiotic
        subfolders.

    Returns
    -------
    list[str]
        Filepaths to selected CSV files.
    list[str]
        Corresponding biotic/abiotic labels.
    """
    # Parse sample text file:
    selected = {"biotic": [], "abiotic": []}
    current_label = None
    # Assign biotic/abiotic labels to each sample name:
    with open(sample_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "# biotic":
                current_label = "biotic"
            elif line == "# abiotic":
                current_label = "abiotic"
            elif current_label is not None:
                selected[current_label].append(line)
    # Match sample names to filepaths in data folders:
    filepaths, labels = [], []

    for label in ["biotic", "abiotic"]:
        folder_path = data_dir + "/" + label
        all_files = os.listdir(folder_path)

        for sample_name in selected[label]:
            matched = [
                f
                for f in all_files
                if f.replace(".CSV", "").replace(".csv", "") == sample_name
            ]
            if matched:
                filepaths.append(folder_path + "/" + matched[0])
                labels.append(label)
            else:
                print(f"Warning: {sample_name} not found in {folder_path}/")

    return filepaths, labels
