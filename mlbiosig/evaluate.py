# ML evaluation/visualisation imports:
from sklearn.inspection import permutation_importance
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, confusion_matrix

# ML interpretability imports:
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from collections import defaultdict
from sklearn.base import clone
import shap

# Misc. Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Visual evaluation metrics:

# Confusion matrices:


def plot_confusion_matrices(models, X_test, y_test, le):
    """
    Plots confusion matrices for all four classifiers, which visually showed
    the true vs predicted biotic/abiotic classifications on the testing
    set.

    False positives (abiotic predicted as biotic) will appear in the top-right
    cell, and are thus of particular importance as well as concertn for life
    detection research.

    Parameters
    ----------
    models : dict
        Dictionary of classifier name to fitted Pipeline from
        train_evaluate_classifiers().
    X_test : pandas.DataFrame
        Test feature matrix.
    y_test : numpy.ndarray
        Test labels.
    le : LabelEncoder
        Fitted LabelEncoder from engineer_features() needed to
        assign class names (abiotic : 0, biotic : 1) for axis labels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    classes = le.classes_  # from fitted label encoder

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
            ax=ax, cmap="magma"
        )
        ax.set_title(name)  # classifier title on each tile

    plt.suptitle("Confusion Matrices: All Classifiers", y=1.02)
    plt.tight_layout()
    plt.show()


# ROC curves:


def plot_roc_curves(models, X_test, y_test):
    """
    Plots ROC (Receiver Operating Characteristic) curves for all
    four classifiers on single axes.

    ROC curves plot true positive rate (sensitivity) against false
    positive rate (1 - specificity) across all decision thresholds,
    whereby the diagonal dashed line represents random classifier
    performance. AUC-ROC scores summarise performance, such that
    1.0 is perfect and 0.5 is random.

    Parameters
    ----------
    models : dict
        Dictionary of classifier name to fitted Pipeline.
    X_test : pandas.DataFrame
        Test feature matrix.
    y_test : numpy.ndarray
        Test labels.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)

    ax.plot([0, 1], [0, 1], "k--", label="random classifier")
    ax.set_title("ROC Curves: All Classifiers")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# Interpretability Tools:

# General permutation importance plotting:


def plot_permutation_importance(model, X_test, y_test, model_name, n_top=10, ax=None):
    """
    Plots calculated permutation importance for ML classifier, showing
    the `n_top` m/z features by mean F1 score as they decrease when each
    feature is randomly shuffled.

    Permutation importance measures how much model performance drops
    when a feature's values are randomly permuted. A large score drop means
    that the feature is important to classifier predictions, while a near-zero
    (or negative) drop suggests the feature contributes little information to
    discriminative classification.

    Error bars (whiskers) shows variability across 30 permutation repeats
    (characterised by the `n_repeats` hyperparameter), indicating how stable
    each importance estimate is.

    Parameters
    ----------
    model : fitted scikit-learn estimator
        Best model estimator.
    X_test : pandas.DataFrame
        Test feature matrix.
    y_test : numpy.ndarray
        Test labels.
    model_name : str
        Classifier name for plot title.
    n_top : int
        Number of top features to display (default = 10).
    ax : matplotlib.axes.Axes, optional
        Axes to plot onto. If None, create a new figure.
    """
    # Compute permutation importance on test set:
    perm_imp = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=42, scoring="f1", n_jobs=-1
    )
    # Sort top `n_top` permutated feature importance:
    sorted_idx = perm_imp.importances_mean.argsort()[-n_top:]
    importances = pd.DataFrame(
        perm_imp.importances[sorted_idx].T, columns=X_test.columns[sorted_idx]
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    # Plotting bar plot with whiskers for F1 score decrease:
    importances.plot.box(
        vert=False,
        whis=10,
        patch_artist=True,
        boxprops=dict(facecolor="lightpink", color="deeppink"),
        whiskerprops=dict(color="deeppink"),
        capprops=dict(color="deeppink"),
        medianprops=dict(color="deeppink"),
        ax=ax,
    )
    ax.set_title(f"Permutation Importances: {model_name}")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in F1 Score")
    ax.set_ylabel("m/z")
    ax.figure.tight_layout()


# Unsupervised hierarchical clustering of features for tree-based
# model permutation feature importance:


def cluster_features(X_train, t=0.18):
    """
    Performs hierarchical clustering of m/z features using Spearman
    correlation to group correlated features into clusters.

    In py-MS data, many m/z values co-occur because they derive from
    the same parent compound fragmenting simultaneously. Due to the strong
    multicollinearity of m/z fragment ions in py-MS datasets, permutation
    importance is distributed across correlated features and produces
    near-zero importances. Hierarchical clustering tackles this by grouping
    correlated features and selecting a single representative per cluster
    to be evaluated. Hierarchical clustering groups such correlated features
    so that a single representative per cluster can be evaluated,
    recovering the true discriminative importance of that chemical group.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    t : float
        Distance threshold hyperparameter for cluster formation
        (default = 0.18).

    Returns
    -------
    dict
        Dictionary mapping cluster ID to list of feature indices.
    numpy.ndarray
        Ward's linkage matrix for dendrogram plotting.
    """
    # Creating Spearman correlation matrix to compute cluster distances:
    corr = spearmanr(X_train).correlation
    corr = (corr + corr.T) / 2  # make correlation matrix symmetric
    np.fill_diagonal(corr, 1)  # 1 on diagonal for same feature correlation

    # Perform hierarchical clustering using Ward's linkage
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))  # Ward's linkage

    cluster_ids = hierarchy.fcluster(
        dist_linkage,  # Ward's linkage
        t,  # t hyperparameter tuned using dendogram
        criterion="distance",
    )

    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    return cluster_id_to_feature_ids, dist_linkage


def plot_dendrogram(X_train, dist_linkage, t=0.18):
    """
    Plots a dendrogram of m/z feature hierarchical clustering,
    in order to show which features group together based on Spearman
    correlation distance.

    Dendrograms visualise the hierarchical merging of features
    into clusters, where the y-axis shows the distances at which clusters
    merge and the red dashed line indicates the chosen threshold t.
    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    dist_linkage : numpy.ndarray
        Ward's linkage matrix from cluster_features().
    t : float
        Distance threshold for reference (default = 0.18).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot dendogram of hierarchical relationship between features
    dendro = hierarchy.dendrogram(
        dist_linkage,  # Ward's linkage
        labels=X_train.columns.to_list(),
        ax=ax,
        leaf_rotation=90,
    )
    # Formatting
    ax.axhline(t, color="r", linestyle="--", label=f"t = {t}")
    ax.set_xlabel("m/z:")
    ax.set_ylabel("Cluster Distance:")
    ax.set_title("Dendrogram of m/z Feature Clusters:")
    leaf_positions = np.arange(5, len(dendro["ivl"]) * 10, 10)
    ax.set_xticks(leaf_positions[::10])  # 10th m/z feature
    ax.set_xticklabels(dendro["ivl"][::10], fontsize=10)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def find_cluster_representatives(cluster_id_to_feature_ids, grid_searches, X_train):
    """
    Selects the most important representative m/z feature from each
    hierarchical cluster for two tree-based models (RF and XGBoost)
    based on the model's feature importances.

    For each cluster of correlated m/z features, features with the highest
    importance are selected as their cluster representative. Permutation
    importance is calculated on this smaller and decorrelated feature subset
    in order to evaluate the true importance of each chemical cluster group.

    Parameters
    ----------
    cluster_id_to_feature_ids : dict
        Dictionary for mapping cluster ID to feature indices
        from cluster_features().
    grid_searches : dict
        Dictionary of fitted classifier name from
        train_evaluate_classifiers().
    X_train : pandas.DataFrame
        Training feature matrix.

    Returns
    -------
    pandas.Index
        Selected representative feature names for RF.
    pandas.Index
        Selected representative feature names for XGBoost.
    """
    # Choose reps. with highest Gini importance:
    rf_importances = (
        grid_searches["Random Forest"]
        .best_estimator_.named_steps["classifier"]
        .feature_importances_
    )

    # Choose reps. with highest gain importance:
    xgb_importances = (
        grid_searches["XGBoost"]
        .best_estimator_.named_steps["classifier"]
        .feature_importances_
    )

    rf_rep, xgb_rep = [], []  # representative features per cluster

    for feat_indices in cluster_id_to_feature_ids.values():
        # find feature with max. corresponding importance:
        rf_rep.append(max(feat_indices, key=lambda i: rf_importances[i]))
        xgb_rep.append(max(feat_indices, key=lambda i: xgb_importances[i]))

    return X_train.columns[rf_rep], X_train.columns[xgb_rep]


# Permutation importance for tree-based models:


def plot_permutation_importance_tree(
    grid_searches,
    X_train,
    X_test,
    y_train,
    y_test,
    model="Random Forest",
    t=0.18,
    n_top=10,
    ax=None,
):
    """
    Plots permutation importance for tree-based models (RF or XGBoost)
    using unsupervised hierarchical clustering on m/z features to tackle
    correlated m/z feature groups.

    This tool clusters correlated features, selects one representative
    per cluster based on high Gini/gain importance, refits the model on the
    reduced feature subset, and produces the permutation importance bar plot
    on decorrelated m/z features.

    Parameters
    ----------
    grid_searches : dict
        Dictionary/classifier name to fitted GridSearchCV.
    X_train : pandas.DataFrame
        Training feature matrix.
    X_test : pandas.DataFrame
        Test feature matrix.
    y_train : numpy.ndarray
        Training labels.
    y_test : numpy.ndarray
        Test labels.
    model : str
        Chosen tree-based model (default = "Random Forest").
    t : float
        Hierarchical clustering distance hyperparameter (default = 0.18).
    n_top : int
        Number of top features to display (default = 10).
    ax : matplotlib.axes.Axes, optional
        Axes to plot onto. If None, create a new figure.
    """
    # Check user-input is correct:
    if model not in ["Random Forest", "XGBoost"]:
        raise ValueError("Model must be 'Random Forest' or 'XGBoost'.")

    # Matches feature indices to cluster + calculate Ward's linkage:
    cluster_id_to_feature_ids, dist_linkage = cluster_features(X_train, t=t)

    # Obtains representative features per cluster:
    rf_features, xgb_features = find_cluster_representatives(
        cluster_id_to_feature_ids, grid_searches, X_train
    )

    # Check which tree-based model was chosen:
    if model == "Random Forest":
        selected_features = rf_features
    elif model == "XGBoost":
        selected_features = xgb_features

    # Refit models on cluster-representative features:
    model_clustered = clone(grid_searches[model].best_estimator_)
    model_clustered.fit(X_train[selected_features], y_train)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    plot_permutation_importance(
        model_clustered, X_test[selected_features], y_test, model, n_top=n_top, ax=ax
    )
    fig.tight_layout()
    plt.show()


# Coefficients for linear models:


def linear_model_coefficients(models, X_test, model="Logistic Regression"):
    """
    Obtains linear coefficients (specifically for Logistic Regression
    and SVC models), showing which m/z features most strongly drive biotic
    (+ve coefficient) and abiotic (-ve coefficient) classification.

    SVC coefficients are only available for linear kernels, so if a
    non-linear kernel (i.e., RBF) was selected during hyperparameter
    tuning, SVC coefficients will not be obtained and will return an
    appropriate message as such.

    Parameters
    ----------
    models : dict
        Dictionary of classifier name to fitted Pipeline.
    X_test : pandas.DataFrame
        Test feature matrix.
    model : str
        Chosen model for analysis (default = "Logistic Regression").

    Returns
    -------
    pandas.DataFrame
        Coefficient DataFrame (sorted by absolute value).
    """
    # Check if incorrect model was inputted:
    if model not in ["Logistic Regression", "SVC"]:
        raise ValueError("Model must be 'Logistic Regression' or 'SVC'.")

    classifier = models[model].named_steps["classifier"]

    # Check if SVC kernel is not linear:
    if model == "SVC" and classifier.kernel != "linear":
        print(
            f"SVC kernel is {classifier.kernel}, coefficients"
            f"are not interpretable for non-linear kernels."
        )
        return None

    # Obtaining linear coefficients:
    coef_df = pd.DataFrame(
        {"m/z Feature": X_test.columns, "Coefficient": classifier.coef_[0]}
    ).sort_values("Coefficient", key=abs, ascending=False)

    print(f"{model}:")
    print("     \nTop 10 Biotic Drivers (positive coefficients):")
    print(coef_df[coef_df["Coefficient"] > 0].head(10).to_string())
    print("     \nTop 10 Abiotic Drivers (negative coefficients):")
    print(coef_df[coef_df["Coefficient"] < 0].head(10).to_string())

    return coef_df


# SHAP beeswarm plot for tree-based models:


def plot_shap_beeswarm(models, X_test, model="Random Forest", n=10):
    """
    Computes SHAP values for tree-based models (RF and XGBoost) and
    plots summary 'beeswarm' plots.

    SHAP (SHapley Additive exPlanations) values quantify each
    feature's contribution to individual biotic/abiotic predictions using
    Shapley values from game theory. Positive SHAP values push
    predictions towards biotic, while negative values favour abiotic.
    The beeswarm plot summarises global feature importance across
    all test samples, where each row is an m/z feature, each dot is a
    sample, and the colour encodes the feature's intensity value indicated
    by the colour bar.

    Parameters
    ----------
    models : dict
        Dictionary of classifier name to fitted Pipeline.
    X_test : pandas.DataFrame
        Test feature matrix.
    model : str
        Chosen tree-based model (default = "Random Forest").
    n : int
        Number of top SHAP values to display (default = 10).
    """
    # Check if incorrect model was inputted:
    if model not in ["Random Forest", "XGBoost"]:
        raise ValueError("Model must be 'Random Forest' or 'XGBoost'.")

    # Compute SHAP values:
    if model == "Random Forest":
        print("Computing SHAP values...")
        fitted_rf = models["Random Forest"].named_steps["classifier"]
        # set feature_perturbation due to correlated m/z values:
        rf_explainer = shap.TreeExplainer(
            fitted_rf, feature_perturbation="tree_path_dependent"
        )
        rf_shap_values = rf_explainer(X_test)
        rf_shap_biotic = rf_shap_values[:, :, 1]  # biotic class
        print(f"RF SHAP values shape: {rf_shap_values.shape}")
        print("RF SHAP Beeswarm (biotic class):")
        # Plot beeswarm:
        shap.plots.beeswarm(rf_shap_biotic, max_display=10)
    else:
        # Compute XGBoost SHAP values:
        print("Computing XGBoost SHAP values...")
        fitted_xgb = models["XGBoost"].named_steps["classifier"]
        xgb_explainer = shap.TreeExplainer(
            fitted_xgb, feature_perturbation="tree_path_dependent"
        )
        xgb_shap_values = xgb_explainer(X_test)
        print(f"XGBoost SHAP values shape: {xgb_shap_values.shape}")
        print("XGBoost SHAP Beeswarm:")
        # Plot beeswarm:
        shap.plots.beeswarm(xgb_shap_values, max_display=20)


# SHAP dependence plot for tree-based models:


def plot_shap_dependence(models, X_test, model, shap_features=None, n_cols=2):
    """
    Computes SHAP values for tree-based models (RF and XGBoost) and
    generates dependence scatter plots for user-specified list of
    m/z features that are of chemical interest.

    Dependence scatter plots show how a single m/z feature's SHAP
    value varies with its intensity, coloured by the feature it
    most strongly interacts with. This can show whether a feature's
    contribution is linear, threshold-dependent, or indicative of other
    geochemical behaviours.

    Parameters
    ----------
    models : dict
        Dictionary of classifier name to fitted Pipeline.
    X_test : pandas.DataFrame
        Test feature matrix.
    model : str
        Chosen classifier model name.
    shap_features : list of int, optional
        List of m/z values to plot as SHAP dependence scatter plots.
        If None, only beeswarm plots are shown.
    n_cols : int
        Number of columns for plotting.
    """
    # Check if correct model name is inputted:
    if model not in ["Random Forest", "XGBoost"]:
        raise ValueError("Model name must be 'Random Forest' or 'XGBoost'.")

    # SHAP values:
    print("Computing SHAP values...")
    fitted_model = models[model].named_steps["classifier"]

    # set feature_perturbation due to correlated m/z values:
    model_explainer = shap.TreeExplainer(
        fitted_model, feature_perturbation="tree_path_dependent"
    )
    shap_values = model_explainer(X_test)

    # Dependence plots for inputted features:
    if shap_features is not None:
        shap_summary = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=X_test.values,
            feature_names=[str(x) for x in X_test.columns],
        )
        # Construct axes according to number of m/z features:
        n = len(shap_features)
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 4 * n_rows))
        axes = axes.ravel()
        # Show dependence plots:
        for ax, feature in zip(axes, shap_features):
            shap.plots.scatter(
                shap_summary[:, str(feature)],
                color=shap_summary,
                ax=ax,
                show=False,
                alpha=0.7,
            )
            ax.set_title(f"m/z {feature}")
        # Hide unused axes:
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle(f"{model} SHAP Dependence Plots: Selected m/z Features")
        plt.tight_layout()
        plt.show()
