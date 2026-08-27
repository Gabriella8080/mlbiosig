# Preprocessing module:
from .preprocess import samples_labels, build_features  # noqa: F401

# Feature engineering module:
from .feature_engineering import (  # noqa: F401
    prepare_features_target,
    encode_labels,
    remove_contaminants,
    apply_var_threshold,
    scale_features,
    classify_samples,
    plot_loadings,
    engineer_features,
)

from .classifiers import grid_search_params, train_evaluate_classifiers  # noqa: F401
from .evaluate import (  # noqa: F401
    plot_confusion_matrices,
    plot_roc_curves,
    plot_permutation_importance,
    plot_dendrogram,
    cluster_features,
    find_cluster_representatives,
    plot_permutation_importance_tree,
    linear_model_coefficients,
    plot_shap_beeswarm,
    plot_shap_dependence,
)
