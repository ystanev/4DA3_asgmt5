from numpy import *

rawTrainingData = loadtxt('data1.txt', delimiter=',')  # read '.txt' file
# print rawTrainingData

X = rawTrainingData[:, 0:2]  # attributes
y = rawTrainingData[:, 2]  # class
# print x, y


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=3, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """

        rgen =random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            print ("Weights: ", "\t", self.w_)  # output the weighs for each iteration
        return self

    def net_input(self, X):
        """Calculate net input"""
        return dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return where(self.net_input(X) >= 0.0, 1, -1)


perceptron = Perceptron()  # creates an object from class Perceptron()

perceptron.fit(X, y)
print ("\n", "Class: ", perceptron.predict(X))
