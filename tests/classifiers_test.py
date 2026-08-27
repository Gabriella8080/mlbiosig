from mlbiosig import grid_search_params


# Create class to make an object where best parameters can
# be accessed during tests checking hyperparameter tuning:
class CheckSearch:
    def __init__(self, best_params):
        self.best_params_ = best_params


# Testing GridSearchCV handling of max_depth = None:


def test_grid_search_none():
    rand_results = {
        "Logistic Regression": CheckSearch(
            {"classifier__C": 1, "classifier__solver": "lbfgs"}
        ),
        "SVC": CheckSearch(
            {
                "classifier__C": 1,
                "classifier__kernel": "rbf",
                "classifier__gamma": "scale",
            }
        ),
        "Random Forest": CheckSearch(
            {
                "classifier__n_estimators": 100,
                "classifier__max_depth": None,
                "classifier__min_samples_split": 5,
            }
        ),
        "XGBoost": CheckSearch(
            {
                "classifier__learning_rate": 0.3,
                "classifier__n_estimators": 100,
                "classifier__max_depth": None,
            }
        ),
    }

    grid = grid_search_params(rand_results)

    # Checking RF and XGBoost max_depth handling:
    assert grid["Random Forest"]["classifier__max_depth"] == [None, 5, 10, 20]
    assert grid["XGBoost"]["classifier__max_depth"] == [None, 5, 10, 20]


# Checking hyperparameters are positive & within valid range:


def test_grid_search_ranges():
    rand_results = {
        "Logistic Regression": CheckSearch(
            {"classifier__C": 1, "classifier__solver": "lbfgs"}
        ),
        "SVC": CheckSearch(
            {
                "classifier__C": 1,
                "classifier__kernel": "rbf",
                "classifier__gamma": "scale",
            }
        ),
        "Random Forest": CheckSearch(
            {
                "classifier__n_estimators": 100,
                "classifier__max_depth": None,
                "classifier__min_samples_split": 5,
            }
        ),
        "XGBoost": CheckSearch(
            {
                "classifier__learning_rate": 0.3,
                "classifier__n_estimators": 100,
                "classifier__max_depth": None,
            }
        ),
    }

    grid = grid_search_params(rand_results)
    rf = grid["Random Forest"]
    xgb = grid["XGBoost"]

    # Checking minimum RF hyperparameters set in tuning are not violated:
    assert all(n >= 50 for n in rf["classifier__n_estimators"])
    assert all(m is None or m >= 1 for m in rf["classifier__max_depth"])
    assert all(d is None or d >= 1 for d in rf["classifier__min_samples_split"])

    # Checking minimum XGBoost hyperparameters:
    assert all(n >= 50 for n in xgb["classifier__n_estimators"])
    assert all(m is None or m >= 1 for m in xgb["classifier__max_depth"])
