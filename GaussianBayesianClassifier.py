import numpy as np

class GaussianBC:
    def __init__(self):
        self.numOfClasses = None
        self.numOfFeatures = None
        self.means = None
        self.cov = None
        self.numOfDataOfeachClass = None
        self.numOftrainingData = np.zeros(1)

    def __getInfoFromDataset(self, separated):
        """
            The function get some information from the separated dataset

            args: 
                separated: dictionary object where each key is target label(class value) and then add a list of all the records as the value in the dictionary
        """
        self.numOfClasses = len(separated.keys())  # Get the number of class
        for i in range(self.numOfClasses):
            if len(separated[i]) != 0:
                self.numOfFeatures = len(separated[i][0])
                break

        self.numOfDataOfeachClass = np.zeros(self.numOfClasses)
        for i in range(self.numOfClasses):
            numOfDataOfClass = len(separated[i])
            self.numOfDataOfeachClass[i] = numOfDataOfClass
            self.numOftrainingData += numOfDataOfClass

    def __separate_by_class(self, X, Y):
        """
            This function split the dataset by class values, return dictionary

            args: 
                X: training data
                Y: target label(class value)
            return:
                separated: dictionary object where each key is target label(class value) and then add a list of all the records as the value in the dictionary
        """

        separated = dict()
        for i in range(len(X)):
            vector = X[i]
            class_value = Y[i]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)

        self.__getInfoFromDataset(separated)

        return separated

    def __summarize_dataset(self, separated):
        """
            Calculate the mean, cov and count for each column(feature) in separated dataset

            args:
                separated: dictionary object where each key is target label(class value) and then add a list of all the records as the value in the dictionary

            returns: 
                means   : The means of each class
                cov: The covariance maxtirx of each class 
        """

        means = np.zeros(shape=(self.numOfClasses, self.numOfFeatures))
        cov = np.zeros(shape=(self.numOfClasses, self.numOfFeatures, self.numOfFeatures))
        for class_value, rows in separated.items():
            # The mean for each input feature
            means[class_value] = np.mean(rows, axis=0)
            cov[class_value] = np.cov(np.array(rows).T)

        return means, cov

    def __multivariate_gaussian_pdf(self, X, mean, cov):
        """
            Returns the pdf of a multivariate gaussian distribution
        """

        cov_inv = np.linalg.inv(cov)
        denominator = np.sqrt(((2 * np.pi)**self.numOfFeatures) * np.linalg.det(cov))
        exponent = -(1/2) * ((X - mean) @ cov_inv @ (X - mean).T)

        return (1 / denominator) * np.exp(exponent)

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

        X = np.array(X)
        y = np.array(y)
        separated = self.__separate_by_class(X, y)
        self.means, self.cov = self.__summarize_dataset(separated)

    def predict(self, X):
    
        numOfTest = X.shape[0]
        best_labels = np.full(numOfTest, -1)
        # print("The Number of test data : ", numOfTest)
        for i in range(numOfTest):
            best_label, best_prob = None, -np.inf
            for j in range(self.numOfClasses):
                probability = (self.numOfDataOfeachClass[j] / self.numOftrainingData)
                probability *= self.__multivariate_gaussian_pdf(X[i], self.means[j], self.cov[j])

                if (best_prob < probability):
                    best_prob = probability
                    best_label = j
            best_labels[i] = best_label
        return best_labels

    def predict_proba(self, X):
        
        X = np.atleast_2d(X) # numpy array and make sure at least two dimenison

        numOfTest = X.shape[0]

        probs = np.full((numOfTest, self.numOfClasses), np.inf)

        for i in range(numOfTest):
            for j in range(self.numOfClasses):
                probability = (self.numOfDataOfeachClass[j] / self.numOftrainingData)
                probability *= self.__multivariate_gaussian_pdf(X[i], self.means[j], self.cov[j])

                probs[i, j] = probability

        # Normalization, each element divide by sum of all elements
        for i in range(numOfTest):
            row = probs[i]
            total_prob = np.sum(row)
            # print(total_prob)
            probs[i] = row / total_prob

        return probs

    def score(self, X, Y):
        """
            Return the  accuracy on the given test data and labels.

            args: 
                X: test data
                Y: ground-truth labels

            return:
                accuracy
        """
        X = np.array(X)  # translate testing data to numpy array

        count_correct = (self.predict(X) == Y).sum()
        return count_correct / X.shape[0]

