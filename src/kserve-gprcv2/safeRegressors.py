from sklearn.base import BaseEstimator, RegressorMixin
from numpy.linalg import LinAlgError
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import signal
from sklearn.utils import shuffle
from threading import Thread
import time
# Create a class to handle timeout situations
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException



class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

class SafeSVR(BaseEstimator, RegressorMixin):
    def __init__(self, C, kernel, gamma, epsilon, timeout):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.epsilon = epsilon
        self.timeout = timeout

    def _create_svr(self):
        return SVR(C=self.C, kernel=self.kernel, gamma=self.gamma, epsilon=self.epsilon)

    def _fit_with_timeout(self, X, y, result):
        try:
            self.svr.fit(X, y)
            result.append(True)
        except Exception as e:
            result.append(e)

    def fit(self, X, y):
        self.svr = self._create_svr()
        result = []

        thread = Thread(target=self._fit_with_timeout, args=(X, y, result))
        thread.start()

        thread.join(timeout=self.timeout)

        if not result:
            raise TimeoutException("Fitting has been interrupted due to timeout.")
        elif isinstance(result[0], Exception):
            raise result[0]

        return self

    def predict(self, X):
        y_pred = self.svr.predict(X)
        return y_pred

    def get_params(self, deep=True):
        return {'C': self.C, 'kernel': self.kernel, 'gamma': self.gamma, 'epsilon': self.epsilon, 'timeout': self.timeout}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.svr = self._create_svr()
        return self



# class SafeSVR(BaseEstimator, RegressorMixin):
#     def __init__(self, C, kernel, gamma, epsilon, timeout):
#         self.C = C
#         self.kernel = kernel
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.timeout = timeout

#     def _create_svr(self):
#         return SVR(C=self.C, kernel=self.kernel, gamma=self.gamma, 
#             epsilon=self.epsilon)

#     def fit(self, X, y):
#         self.svr = self._create_svr()

#         # Setting timeout signal
#         signal.signal(signal.SIGALRM, timeout_handler)
#         signal.alarm(self.timeout)
#         try:
#             # shuffle the data
#             print('starting fitting')
#             self.svr.fit(X, y)
#             print('ending fitting')
#             signal.alarm(0)
#         except TimeoutException:
#             raise TimeoutException("Fitting has been interrupted due to timeout.")
#         except LinAlgError:
#             raise LinAlgError("Linear Algebra calculation failed.")
#         return self

#     def predict(self, X):
#         # Setting timeout signal
#         signal.signal(signal.SIGALRM, timeout_handler)
#         signal.alarm(self.timeout)
#         try:
#             print('starting prediction')
#             y_pred = self.svr.predict(X)
#             print('ending prediction')
#             signal.alarm(0)
#         except TimeoutException:
#             raise TimeoutException("Prediction has been interrupted due to timeout.")
#         except LinAlgError:
#             raise LinAlgError("Linear Algebra calculation failed.")
#         return y_pred

#     def get_params(self, deep=True):
#         return {'C': self.C, 'kernel': self.kernel, 
#             'gamma': self.gamma, 'epsilon': self.epsilon, 'timeout': self.timeout}

#     def set_params(self, **parameters):
#         for parameter, value in parameters.items():
#             setattr(self, parameter, value)
#         self.svr = self._create_svr()
#         return self

class SafeRandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=42, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='auto', bootstrap=True, max_leaf_nodes=None,
                 timeout=120):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.timeout = timeout

    def _create_rf(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=42
        )

    def fit(self, X, y):
        self.rf = self._create_rf()
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)
        try:
            self.rf.fit(X, y)
            signal.alarm(0)
        except TimeoutException:
            raise TimeoutException("Fitting has been interrupted due to timeout.")
        except LinAlgError:
            raise LinAlgError("Linear Algebra calculation failed.")
        return self

    def predict(self, X):
        return self.rf.predict(X)

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'max_leaf_nodes': self.max_leaf_nodes,
            'timeout': self.timeout
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.rf = self._create_rf()
        return self
      
      


class SafeXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1):
        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def _create_xgb(self):
        return XGBRegressor(
            objective=self.objective,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate
        )

    def fit(self, X, y):
        self.xgb = self._create_xgb()
        try:
            self.xgb.fit(X, y)
        except LinAlgError:
            print("LinAlgError encountered. Using default XGBRegressor model.")
            self.objective = 'reg:squarederror'
            self.n_estimators = 100
            self.max_depth = 3
            self.learning_rate = 0.1
            self.xgb = self._create_xgb()
            self.xgb.fit(X, y)
        return self

    def predict(self, X):
        return self.xgb.predict(X)

    def get_params(self, deep=True):
        return {
            'objective': self.objective,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.xgb = self._create_xgb()
        return self





class SafeLGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=-1, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def _create_lgbm(self):
        return LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate
        )

    def fit(self, X, y):
        self.lgbm = self._create_lgbm()
        try:
            self.lgbm.fit(X, y)
        except LinAlgError:
            print("LinAlgError encountered. Using default LGBMRegressor model.")
            self.n_estimators = 100
            self.max_depth = -1
            self.learning_rate = 0.1
            self.lgbm = self._create_lgbm()
            self.lgbm.fit(X, y)
        return self

    def predict(self, X):
        return self.lgbm.predict(X)

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.lgbm = self._create_lgbm()
        return self
