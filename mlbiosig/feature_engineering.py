# Imports
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, RobustScaler
import matplotlib.pyplot as plt
import pandas as pd

# Contaminated m/z features:
contaminants = [
    18,
    28,
    32,
    44,  # atmospheric/background
    77,
    94,
    115,
    141,
    168,  # diffusion pump fluid
    170,
    262,
    354,
    446,  # diffusion pump fluid
    73,
    147,
    207,
    221,  # GC column bleed
    281,
    295,
    355,
    429,  # GC column bleed
]


# Separating target and features
def prepare_features_target(df):
    """
    Separates feature matrix into features (X) and target (y),
    then applies spectral binning to reduce fractional m/z columns
    into integer m/z bins.

    Parameters
    ----------
    df : pandas.DataFrame
        Stacked py-MS feature matrix from build_features(), with
        m/z columns and label column ('biotic'/'abiotic class').

    Returns
    -------
    pandas.DataFrame
        Binned feature matrix X with integer m/z columns.
    pandas.Series
        Target label series y ('biotic'/'abiotic').
    """
    # Separating target (y) from features (X):
    y = df["label"]  # target
    X = df.drop(columns=["label"])  # features

    # Round fractional m/z column names to nearest integer
    X.columns = X.columns.astype(float).round(0).astype(int)

    # Transpose rows + sum intensities of columns within integer bins
    X_binned = X.T.groupby(level=0).sum().T

    return X_binned, y


# Encoding categorical target
def encode_labels(y):
    """
    Encodes str 'biotic'/'abiotic' labels into integers
    for ML binary classification, such that:
        - 'abiotic' = 0,
        - 'biotic' = 1.

    Parameters
    ----------
    y : pandas.Series
        Target label series ('biotic'/'abiotic').

    Returns
    -------
    numpy.ndarray
        Encoded integer label array.
    LabelEncoder
        Fitted LabelEncoder.
    """
    # Initialise LabelEncoder:
    le = LabelEncoder()

    # Fit and convert categorical labels into integers:
    y_encoded = le.fit_transform(y)

    return y_encoded, le


# Removing contaminants & artefacts
def remove_contaminants(X_train, X_test, contaminants=contaminants):
    """
    Removes documented contaminant m/z features from training and
    testing feature sets.

    Contaminated m/z features range from atmospheric background
    conditions, diffusion pump fluid, and GC column bleed from the
    GC-MS instrument itself.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    X_test : pandas.DataFrame
        Testing feature matrix.
    contaminants : list[int], optional
        List of m/z features to drop (defaults to contaminants).

    Returns
    -------
    pandas.DataFrame
        Cleaned training feature matrix.
    pandas.DataFrame
        Cleaned testing feature matrix.
    """
    # Check if m/z value exists as a feature column in sample set:
    contam_cols = [c for c in contaminants if c in X_train.columns]

    # Drop contaminated m/z features from training & testing set:
    X_train_clean = X_train.drop(columns=contam_cols)
    X_test_clean = X_test.drop(columns=contam_cols)

    return X_train_clean, X_test_clean


# Reducing noise using variance threshold
def apply_var_threshold(X_train, X_test, threshold=0.001):
    """
    Removes low-variance m/z features from training and testing
    feature sets using a variance threshold fitted on training
    data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Cleaned training feature matrix.
    X_test : pandas.DataFrame
        Cleaned testing feature matrix.
    threshold : float, optional
        Minimum variance for a feature to be kept,
        set at default to 0.001.

    Returns
    -------
    pandas.DataFrame
        Retained training feature matrix.
    pandas.DataFrame
        Retained testing feature matrix.
    VarianceThreshold
        Fitted VarianceThreshold selector.
    """
    # Initialise VarianceThreshold selector:
    selector = VarianceThreshold(threshold=threshold)

    # Remove features where variance is below threshold
    X_train_selected = selector.fit_transform(X_train)  # fit on training
    X_test_selected = selector.transform(X_test)  # apply transform to testing

    # Convert retained features back to dataframe with column names:
    high_var_cols = X_train.columns[selector.get_support()]
    X_train_reduced = pd.DataFrame(
        X_train_selected, columns=high_var_cols, index=X_train.index
    )
    X_test_reduced = pd.DataFrame(
        X_test_selected, columns=high_var_cols, index=X_test.index
    )

    return X_train_reduced, X_test_reduced, selector


# Scaling features
def scale_features(X_train, X_test):
    """
    Applies RobustScaler to training and testing features.

    RobustScaler is chosen instead of StandardScaler as it is
    robust to outlier m/z intensities, which in py-MS data may
    represent a true chemical signal rather than noise.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Retained training feature matrix.
    X_test : pandas.DataFrame
        Retained testing feature matrix.

    Returns
    -------
    pandas.DataFrame
        Scaled training feature matrix.
    pandas.DataFrame
        Scaled testing feature matrix.
    RobustScaler
        Fitted RobustScaler.
    """
    # Initialise RobustScaler:
    scaler = RobustScaler()

    # Fit scaler to training set:
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )

    # Transform fitted scaler on testing set:
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


# Classifying samples for PCA:
def classify_samples(name, cats):
    """
    Helper function for classifying samples into
    provided categories, by matching substrings in
    `cats` to sample name and returning category label.

    Parameters
    ----------
    name : str
        Sample name to be classified.
    cats : dict
        Dictionary where keys are matched with substrings
        in `name`, and values are class labels.

    Returns
    -------
    str
        Matched category label for the sample, or 'Other'
        if no match is found.
    """
    for key, cat in cats.items():
        # for multiple categories listed
        if isinstance(key, tuple):
            if any(sub in name for sub in key):
                return cat
        # for single category
        else:
            if key in name:
                return cat
    # if sample name is not in key + has no category
    return "Other"  # category = 'Other'


# Plotting loadings to visualise m/z feature effect on PCA:
def plot_loadings(pc_loadings, pc_name, n=15):
    """
    Helper function for plotting principal component (PC)
    loadings to understand PC-space clustering.

    Parameters
    ----------
    pc_loadings : pandas.DataFrame
        Dictionary with pre-sorted loadings of a specified
        PC i.e., PC1.
    pc_name : str
        Name of the PC for plotting purposes.
    n : int
        Number of largest and smallest loadings to be plotted
        for a given PC (default = 15).
    """
    # Obtain largest and smallest 15 loadings
    top = pd.concat([pc_loadings.nsmallest(n), pc_loadings.nlargest(n)])
    # Rank loadings in descending order (top -> bottom ordering):
    top = top.reindex(top.abs().sort_values(ascending=False).index)
    colours = [
        "indigo" if x < 0 else "deeppink" for x in top
    ]  # pink = +ve, purple = -ve

    # Plot horizontal bar chart:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(top.index.astype(str), top.values, color=colours, edgecolor="none")

    # Formatting:
    ax.spines["top"].set_visible(False)  # remove plotting box
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)  # remove tick marks
    ax.set_xlabel("Loadings:")
    ax.set_ylabel("m/z:")
    ax.set_title(f"Top {n} {pc_name} Loadings:")
    plt.tight_layout()
    plt.show()


# Main function performing all feature engineering steps
def engineer_features(
    df,
    test_size=0.2,
    random_state=42,
    contaminants=contaminants,
    variance_threshold=0.001,
):
    """
    Main function for full feature engineering pipeline, transforming
    raw feature matrix to split, cleaned, scaled, and ML-ready training
    and testing sets.

    Feature engineering chronologically includes m/z spectral binning,
    target label encoding, train/test split, m/z contaminant removal,
    variance thresholding, and RobustScaler normalisation.

    Parameters
    ----------
    df : pandas.DataFrame
        Stacked py-MS feature matrix from build_features().
    test_size : float
        Proportion of dataset for test split (default = 0.2).
    random_state : int
        Random seed (default = 42).
    contaminants : list[int]
        m/z values to remove as contaminants (default set to
        'contaminants').
    variance_threshold : float
        Minimum variance for features (default = 0.001).

    Returns
    -------
    pandas.DataFrame
        Scaled training feature matrix.
    pandas.DataFrame
        Scaled testing feature matrix.
    numpy.ndarray
        Training labels.
    numpy.ndarray
        Testing labels.
    LabelEncoder, VarianceThreshold
        Fitted LabelEncoder and VarianceThreshold selector.
    """
    # Separating features & target, followed by feature spectral binning:
    X_binned, y = prepare_features_target(df)

    # Encoding target from 'biotic/abiotic' labels to numerical data:
    y_encoded, le = encode_labels(y)

    # Train/test split, with dataset shuffled to break sequential
    # biotic-abiotic block order of features:
    X_train, X_test, y_train, y_test = train_test_split(
        X_binned,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y_encoded,
    )

    # Remove contaminated m/z features from train + test:
    X_train_clean, X_test_clean = remove_contaminants(X_train, X_test, contaminants)

    # Apply variance thresholding to train + test:
    X_train_reduced, X_test_reduced, selector = apply_var_threshold(
        X_train_clean, X_test_clean, variance_threshold
    )

    return X_train_reduced, X_test_reduced, y_train, y_test, le, selector
