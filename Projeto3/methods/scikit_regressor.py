from sklearn import linear_model
from sklearn.metrics import accuracy_score


class ScikitRegressor:
    @staticmethod
    def ovr_regressor(train_x, train_y, max_iterations, tolerance):
        return ScikitRegressor.multiclass_regressor('ovr', train_x, train_y, max_iterations, tolerance)

    @staticmethod
    def multinomial_regressor(train_x, train_y, max_iterations, tolerance):
        return ScikitRegressor.multiclass_regressor('multinomial', train_x, train_y, max_iterations, tolerance)

    @staticmethod
    def multiclass_regressor(multiclass, train_x, train_y, max_iterations, tolerance):
        model = linear_model.LogisticRegression(multi_class=multiclass, solver='sag', max_iter=max_iterations
                                                             , tol=tolerance)
        # Fit model
        model.fit(train_x, train_y)

        return model
