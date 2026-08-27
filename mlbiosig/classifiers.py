# Model imports:
from sklearn.linear_model import LogisticRegression  # LR
from sklearn.ensemble import RandomForestClassifier  # RF
from sklearn.svm import SVC  # SVC
from xgboost import XGBClassifier  # XGB

# ML imports:
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

# Misc. imports:
from scipy.stats import randint
from tqdm.auto import tqdm
import warnings  # hide warnings

# Ignore FutureWarning from model hyperparameter tuning:
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")

# Stratified CV strategy due to high imbalance:
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Building classifier pipelines:


def build_pipelines(imbalance_ratio):
    """
    Builds pipelines for four classifiers, with scaling and
    class imbalance handling where appropriate.

    Construct sklearn `Pipeline` for Logistic Regression, SVC,
    Random Forest, and XGBoost model, incorporating RobustScaler
    or StandardScaler and class weighting hyperparameters.

    Parameters
    ----------
    imbalance_ratio : float
        Ratio of abiotic:biotic samples in training set,
        used as scale_pos_weight for XGBoost.

    Returns
    -------
    dict
        Dictionary of classifier name to unfitted Pipeline.
    """
    pipelines = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", random_state=42, max_iter=2000
                    ),
                ),
            ]
        ),
        "SVC": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        class_weight="balanced",
                        random_state=42,
                        max_iter=2000,
                        probability=True,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", RobustScaler()),
                (
                    "classifier",
                    RandomForestClassifier(class_weight="balanced", random_state=42),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("scaler", RobustScaler()),
                (
                    "classifier",
                    XGBClassifier(
                        scale_pos_weight=imbalance_ratio,
                        random_state=42,
                        eval_metric="logloss",  # binary cross entropy
                    ),
                ),
            ]
        ),
    }

    return pipelines


# Hyperparameter Tuning:


def random_search_params():
    """
    Defines coarse-tuning RandomSearchCV hyperparameter
    search spaces for all four classifiers (Logistic Regression,
    SVC, Random Forest, XGBoost). The hyperparameter values are
    hard-coded due to prior exploration in notebooks 03 & 04.

    Returns
    -------
    dict
        Dictionary of classifier name to random-search
        parameter grid.
    """
    random_search_spaces = {
        "Logistic Regression": {
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["lbfgs", "liblinear"],
        },
        "SVC": {
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"],
        },
        "Random Forest": {
            "classifier__n_estimators": randint(50, 500),
            "classifier__max_depth": [None, 5, 10, 15, 20, 30, 50],
            "classifier__min_samples_split": randint(5, 20),
        },
        "XGBoost": {
            "classifier__n_estimators": randint(50, 500),
            "classifier__learning_rate": [0.001, 0.01, 0.1, 0.3, 0.5],
            "classifier__max_depth": [None, 5, 10, 15, 20, 30, 50],
        },
    }

    return random_search_spaces


def grid_search_params(random_search_results):
    """
    Defines fine-tuning GridSearchCV hyperparameter
    search spaces for all four classifiers (Logistic Regression,
    SVC, Random Forest, XGBoost). The hyperparameter values are
    hard-coded due to prior exploration in notebooks 03 & 04.

    Parameters
    ----------
    random_search_results : dict
        Dictionary of classifier name to fitted
        RandomizedSearchCV object.

    Returns
    -------
    dict
        Dictionary of classifier name to grid-search
        parameter grid.
    """
    # Extract best hyperparameter combinations per model:
    lr_best = random_search_results["Logistic Regression"].best_params_
    svc_best = random_search_results["SVC"].best_params_
    rf_best = random_search_results["Random Forest"].best_params_
    xgb_best = random_search_results["XGBoost"].best_params_

    # Extract each best hyperparameter:
    best_C_lr = lr_best["classifier__C"]  # LR
    best_solver_lr = lr_best["classifier__solver"]
    best_C_svc = svc_best["classifier__C"]  # SVC
    best_kernel_svc = svc_best["classifier__kernel"]
    best_gamma_svc = svc_best["classifier__gamma"]
    best_n_rf = rf_best["classifier__n_estimators"]  # RF
    best_depth_rf = rf_best["classifier__max_depth"]
    best_minsamples_rf = rf_best["classifier__min_samples_split"]
    best_lrate_xgb = xgb_best["classifier__learning_rate"]  # XGB
    best_n_xgb = xgb_best["classifier__n_estimators"]
    best_depth_xgb = xgb_best["classifier__max_depth"]

    # Handle RF tree depth when = None:
    if best_depth_rf is None:
        rf_depth_grid = [None, 5, 10, 20]
    else:
        rf_depth_grid = [  # max() to prevent > 1
            max(1, best_depth_rf - 5),
            max(1, best_depth_rf - 4),
            max(1, best_depth_rf - 3),
            best_depth_rf,
        ]
    # Handle XGB tree depth when = None:
    if best_depth_xgb is None:
        xgb_depth_grid = [None, 5, 10, 20]
    else:
        xgb_depth_grid = [
            max(1, best_depth_xgb - 5),
            max(1, best_depth_xgb - 4),
            max(1, best_depth_xgb - 3),
            best_depth_xgb,
        ]

    # Building grid search spaces based on best random search results:
    grid_search_spaces = {
        "Logistic Regression": {
            "classifier__C": [
                best_C_lr * 0.1,
                best_C_lr * 0.5,
                best_C_lr,
                best_C_lr * 2,
                best_C_lr * 10,
            ],
            "classifier__solver": [best_solver_lr],
        },
        "SVC": {
            "classifier__C": [
                best_C_svc * 0.1,
                best_C_svc * 0.5,
                best_C_svc,
                best_C_svc * 2,
                best_C_svc * 10,
            ],
            "classifier__kernel": [best_kernel_svc],
            "classifier__gamma": [best_gamma_svc],
        },
        "Random Forest": {
            "classifier__n_estimators": [
                max(50, best_n_rf - 100),  # prevent > 0
                best_n_rf,
                best_n_rf + 100,
            ],
            "classifier__max_depth": rf_depth_grid,
            "classifier__min_samples_split": [
                max(2, best_minsamples_rf - 2),
                best_minsamples_rf,
                best_minsamples_rf + 2,
            ],
        },
        "XGBoost": {
            "classifier__learning_rate": [
                best_lrate_xgb * 0.5,
                best_lrate_xgb,
                best_lrate_xgb * 0.8,
            ],
            "classifier__n_estimators": [
                max(50, best_n_xgb - 100),  # prevent > 0
                best_n_xgb,
                best_n_xgb + 100,
            ],
            "classifier__max_depth": xgb_depth_grid,
        },
    }

    return grid_search_spaces


def format_params(params):
    """
    Remove pipeline prefixes from scikit-learn parameters
    for cleaner outputs during execution.

    Parameters
    ----------
    params : dict
        Dictionary of pipeline hyperparameters returned
        during searches (RandomizedSearchCV, GridSearchCV).

    Returns
    -------
    dict
        Dictionary of hyperparameters with pipeline
        prefixes removed for better readability.
    """
    return {key.replace("classifier__", ""): value for key, value in params.items()}


def do_random_search(pipelines, random_params, X_train, y_train, cv=cv):
    """
    Run coarse RandomizedSearchCV for all four classifiers.

    Parameters
    ----------
    pipelines : dict
        Dictionary of classifier name to Pipeline.
    random_params : dict
        Dictionary of classifier name to coarse
        random-search parameter space.
    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : numpy.ndarray
        Training labels.
    cv : StratifiedKFold
        Cross-validation.

    Returns
    -------
    dict
        Dictionary of classifier name to fitted
        RandomizedSearchCV object.
    """
    random_search_results = {}
    search_summary = []  # for output readability

    for idx, (name, pipeline) in enumerate(
        tqdm(
            pipelines.items(), total=len(pipelines), desc="Random Search", unit="model"
        ),
        start=1,
    ):
        print(f"\n[{idx}/4] {name}")
        search = RandomizedSearchCV(
            pipeline,
            random_params[name],
            cv=cv,
            scoring="f1",  # F1 score
            n_jobs=-1,  # use all CPUs
        )
        search.fit(X_train, y_train)
        random_search_results[name] = search
        # Outputting & saving results:
        clean_params = format_params(search.best_params_)
        print(f"    Best CV F1: {search.best_score_:.3f}")
        print(f"    Parameters: {clean_params}")
        search_summary.append(
            {
                "Search": "Random Search",
                "Model": name,
                "CV F1": search.best_score_,
                "Parameters": clean_params,
            }
        )

    return random_search_results


def do_grid_search(pipelines, grid_params, X_train, y_train, cv=cv):
    """
    Run fine GridSearchCV for all four classifiers.

    Parameters
    ----------
    pipelines : dict
        Dictionary of classifier name to Pipeline.
    grid_params : dict
        Dictionary of classifier name to fine grid-search
        parameter space.
    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : numpy.ndarray
        Training labels.
    cv : StratifiedKFold
        Cross-validation.

    Returns
    -------
    dict
        Dictionary of classifier name to fitted
        GridSearchCV object.
    """
    grid_search_results = {}
    search_summary = []

    for idx, (name, pipeline) in enumerate(
        tqdm(pipelines.items(), total=len(pipelines), desc="Grid Search", unit="model"),
        start=1,
    ):
        print(f"\n[{idx}/4] {name}")
        search = GridSearchCV(
            pipeline,
            grid_params[name],
            cv=cv,
            scoring="f1",  # F1 score
            n_jobs=-1,  # use all CPUs
        )
        search.fit(X_train, y_train)
        grid_search_results[name] = search
        # Outputting & saving results:
        clean_params = format_params(search.best_params_)
        print(f"    Best CV F1: {search.best_score_:.3f}")
        print(f"    Parameters: {clean_params}")
    return grid_search_results


# Evaluate tuned classifiers:


def evaluate_models(grid_search_results, X_test, y_test):
    """
    Evaluate all four tuned classifiers (Logistic Regression, SVC,
    Random Forest, XGBoost) on the testing set. Evaluation metrics
    include CV F1, test F1, and ROC-AUC score.

    Parameters
    ----------
    grid_search_results : dict
        Dictionary of classifier name to fitted
        GridSearchCV object.
    X_test : pandas.DataFrame
        Test feature matrix.
    y_test : numpy.ndarray
        Test labels.

    Returns
    -------
    dict
        Fitted GridSearchCV hyperparameters for
        all four classifiers.
    dict
        Dictionary of classifier name to best fitted
        Pipeline estimator.
    dict
        Dictionary of classifier name and results
        containing CV F1, F1, and ROC-AUC score.
    """
    results = {}
    models = {}

    # Iterate over each classifier, selecting best hyperparameters:
    for name, grid_search in tqdm(
        grid_search_results.items(),
        total=len(grid_search_results),
        desc="Evaluating Models",
        unit="model",
    ):
        model = grid_search.best_estimator_
        y_pred = model.predict(X_test)  # predict class
        y_prob = model.predict_proba(X_test)[:, 1]  # prediction probabilities
        # Collect CV F1, F1, and ROC-AUC score:
        results[name] = {
            "CV_F1": grid_search.best_score_,  # cv f1
            "F1": f1_score(y_test, y_pred),  # f1
            "ROC_AUC": roc_auc_score(y_test, y_prob),  # roc-auc
        }
        models[name] = model
        # Output model evaluation scores:
        print("\n==========================================")
        print(f"\n{name} Results:")
        print(f"    CV F1 Score: {results[name]['CV_F1']:.3f}")
        print(f"    F1 Score: {results[name]['F1']:.3f}")
        print(f"    ROC-AUC Score: {results[name]['ROC_AUC']:.3f}")

    return results, models


def train_evaluate_classifiers(X_train, X_test, y_train, y_test, cv=cv):
    """
    Main function that runs the entire classification pipeline.

    Builds four classifier pipelines (Logistic Regression, SVC, Random
    Forest, XGBoost), runs coarse-to-fine hyperparameter tuning, and
    evaluates classifiers on testing dataset.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    X_test : pandas.DataFrame
        Testing feature matrix.
    y_train : numpy.ndarray
        Training labels.
    y_test : numpy.ndarray
        Testing labels.
    cv : StratifiedKFold
        Cross-validation.

    Returns
    -------
    dict
        Evaluation results for all classifiers.
    dict
        Best fitted Pipeline for all classifiers.
    dict
        Fitted GridSearchCV objects for all classifiers.
    """
    # Class imbalance ratio for XGBoost:
    abiotic_samples = (y_train == 0).sum()  # abiotic encoded as 0
    biotic_samples = (y_train == 1).sum()  # biotic encoded as 1
    imbalance_ratio = abiotic_samples / biotic_samples

    # Build pipelines and parameter spaces:
    pipelines = build_pipelines(imbalance_ratio)
    random_params = random_search_params()

    # Coarse random-search:
    print("\n==========================================")
    print("Coarse Hyperparameter Search: RandomizedSearchCV")
    print("==========================================")
    random_searches = do_random_search(pipelines, random_params, X_train, y_train, cv)

    # Fine grid-search:
    grid_params = grid_search_params(random_searches)
    print("\n==========================================")
    print("Fine Hyperparameter Search: GridSearchCV")
    print("==========================================")
    grid_searches = do_grid_search(pipelines, grid_params, X_train, y_train, cv)

    # Evaluate tuned classifier performance on test set:
    print("\n==========================================")
    print("Test-set Evaluation:")
    print("==========================================")
    results, models = evaluate_models(grid_searches, X_test, y_test)

    return results, models, grid_searches
