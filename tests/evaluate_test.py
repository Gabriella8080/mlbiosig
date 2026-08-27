import pytest
import pandas as pd
from mlbiosig import (
    cluster_features,
    linear_model_coefficients,
    plot_permutation_importance_tree,
)

# Testing ValueError raising in tree-based permutation importance:


def test_tree_permimportance():
    with pytest.raises(ValueError, match="Model must be 'Random Forest' or 'XGBoost'."):
        # Create dummy training set:
        X_train = pd.DataFrame(
            {
                50: [0.1, 0.2, 0.3, 0.4],
                51: [0.2, 0.3, 0.4, 0.1],
                52: [0.3, 0.4, 0.1, 0.2],
            }
        )

        # Test if function allows for non-tree model
        plot_permutation_importance_tree(
            {},  # grid searches
            X_train,
            None,  # X_test
            None,  # y_train
            None,  # y_test
            model="Logistic Regression",  # invalid
        )


# Testing coefficient analysis handling of non-linear SVC:


def test_nonlinear_coefanalysis():
    # Create class to make an object that can be treated as a
    # non-linear RBF SVC model for testing:
    class TestModel:
        def __init__(self):
            self.named_steps = {"classifier": self}
            self.kernel = "rbf"

    models = {"SVC": TestModel()}

    result = linear_model_coefficients(models, None, model="SVC")

    # Check that RBF SVC returns None for coefficient analysis:
    assert result is None


# Test limit of clustering threshold t:


@pytest.mark.parametrize("t", [0.18, 0.5, 10, 100])
def test_t_threshold(t):
    # Set fake intensities value arrays to 4 dummy m/z features:
    X_train = pd.DataFrame(
        {50: [0.1, 0.2, 0.3, 0.4], 51: [0.2, 0.3, 0.4, 0.1], 52: [0.3, 0.4, 0.1, 0.2]}
    )

    clusters, linkage = cluster_features(X_train, t=t)

    # Check there is at least 1 cluster & valid distance matrix shape:
    assert len(clusters) >= 1
    assert linkage.shape[0] == len(X_train.columns) - 1
