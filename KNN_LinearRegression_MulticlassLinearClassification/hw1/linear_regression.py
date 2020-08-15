import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        w = self.weights_
        y_pred = np.dot(X, w)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution
        #  Use only numpy functions. Don't forget regularization.

        w_opt = None
        # ====== YOUR CODE: ======
        N = X.shape[0]
        reg_term = (np.eye(X.shape[1]) * self.reg_lambda * N)
        reg_term[0, 0] = 0.0
        A = np.dot(X.T, X) + reg_term
        b = np.dot(X.T, y)

        w_opt = np.linalg.solve(A, b)

        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None
        # ====== YOUR CODE: ======
        xb = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        polynomial = sklearn.preprocessing.PolynomialFeatures(degree=self.degree)
        X_poly = np.empty((X.shape[0], 5), dtype=X.dtype)

        X_poly[:, 0] = X[:, 13]
        X_poly[:, 1] = X[:, 6]
        X_poly[:, 2] = X[:, 11]
        X_poly[:, 3] = X[:, 3]
        X_poly[:, 4] = X[:, 10]

        X_poly = polynomial.fit_transform(X_poly)

        X_transformed = np.empty((X.shape[0], X_poly.shape[1] + 2))
        X_transformed[:, 0:X_poly.shape[1]] = X_poly
        X_transformed[:, X_poly.shape[1]] = np.sqrt(X[:, 8])
        X_transformed[:, X_poly.shape[1] + 1] = np.sqrt(X[:, 7])
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    corr_matrix = df.corr(method='pearson')
    corr_matrix = corr_matrix.loc[target_feature, :]
    corr_matrix = np.abs(corr_matrix).sort_values(ascending=False)
    corr_matrix = corr_matrix.drop(target_feature)

    top_n_features = corr_matrix.index.values[:n]
    top_n_corr = corr_matrix.iloc[:n].values
    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    mse = np.square(np.subtract(y, y_pred)).mean()
    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    mse = mse_score(y, y_pred)
    mean = mse_score(y,  np.mean(y))

    r2 = 1 - (mse / mean)
    # ========================
    return r2


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    best_score = -float('inf')

    import copy

    cur_params = model.get_params()
    best_params = cur_params.copy()

    mse = 'neg_mean_squared_error'
    r2 = 'r2'

    for degree in degree_range:
        for lamb in lambda_range:

            cur_params['bostonfeaturestransformer'].degree = cur_params['bostonfeaturestransformer__degree'] = degree
            cur_params['linearregressor'].reg_lambda = cur_params['linearregressor__reg_lambda'] = lamb

            model.set_params(**cur_params)

            score = np.mean(sklearn.model_selection.cross_validate(model, X, y, cv=k_folds, scoring=mse)['test_score'])

            if score > best_score:
                best_score = score
                best_params = copy.deepcopy(cur_params)
    # ========================

    return best_params
