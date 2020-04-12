import numpy as np

class linearClassifierGD:
    """
        Learning Regression Using Gradient Desent (Batch Version)

    Parameter
    ------------
    eta: float 
        constant learning rate

    n_iterations: int
        # of passes over the training set


    Attributes
    ------------
    w_ : 
        weights of fitting the model

    cost_: total error of model after each iteration
    """

    def __init__(self,  eta=0.01, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self, X, y):
        """
            The fitting function

            args:
                X : array-like, shape = [n_samples, n_features]
                Training samples
                y : array-like, shape = [n_samples,]
                    Target values
            return:
        """
        # add bias term to X
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        self.cost_ = []
        self.w_ = np.random.rand(X.shape[1])

        m = X.shape[0]
        # gradient desent
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(X.T, residuals)
            self.w_ = self.w_ - (self.eta / m) * gradient_vector
            # record cost in each iteration
            cost = np.sum(residuals ** 2) / (2 * m)

            # Set threshold to stop
            if cost <= 1e-8:  # converge
                print("Linear Classifier has been converge!")
                break
            if cost >= 1e+100: # diverge 
                print("*Linear Classifier has been Diverge!")
                break
            self.cost_.append(cost)

    def predict(self, X):
        """ 
        Predicts the value after the model has been trained.
        args:
            x : array-like, shape = [n_samples, n_features]
                Test samples

        Returns:
            Predicted value
        """

        # add bias term to X
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        return self.__step_function(np.dot(X, self.w_))

    def predict_regressionValue(self, X):
        """ 
        Predicts the value after the model has been trained.
        Predict Regression value, not class label
        args:
            x : array-like, shape = [n_samples, n_features]
                Test samples

        Returns:
            Predicted value
        """

        # add bias term to X
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        return np.dot(X, self.w_)

    def __step_function(self, X):
        """
            Step activiation function used for two class(label:1 and -1) classification with threshold->0
            args:
                X: Predicted value using the linear model
        """
        labels = np.zeros(X.shape[0])

        labels[X >= 0] = 1
        labels[X < 0] = -1

        return labels

    def score(self, X, Y):
        """
            Calculate accuracy(score) used for two class(label:1 and -1) classification with step function
            Note: this function only apply to two class classfication
            args:
                X : array-like, shape = [n_samples, n_features]
                Training samples
                y : array-like, shape = [n_samples,]
                    Target values
            return:
                Accuracy value
        """
        # add bias term to X
        count_correct = (self.__step_function(self.predict(X)) == Y).sum()
        return count_correct / X.shape[0]