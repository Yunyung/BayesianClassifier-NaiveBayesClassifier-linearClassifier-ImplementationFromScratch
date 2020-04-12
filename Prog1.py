import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn import metrics

# my implement modules
from evaluation import Evaluation
from cross_validate import cross_validate
from linearClassifierGD import linearClassifierGD
from GaussianBayesianClassifier import GaussianBC
from GaussianNaiveBayes import GaussianNaiveBayes

def binary_Classification_analysis(X_train, y_train, X_test, y_test):
    BCmodel = GaussianBC()
    NBmodel = GaussianNaiveBayes()
    BCmodel.fit(X_train, y_train)
    NBmodel.fit(X_train, y_train)
    print("Training Data Score-> BC:", BCmodel.score(X_train, y_train), ", NB:", NBmodel.score(X_train, y_train))
    print("Testing Data Score-> BC:", BCmodel.score(X_test, y_test), ", NB:", NBmodel.score(X_test, y_test))
    print("Average Score of Cross-Validation : BC:", np.sum(cv.score(BCmodel, X_train, y_train, 3)) / 3, " NB:", np.sum(cv.score(NBmodel, X_train, y_train, 3)) / 3)


    # (Training Set) Calculate Bayesian Classification confusion Matrix、ROC curve and AUC score, and plot it
    print("Bayesian Classifier confusion Matrix (Training Set):\n", metrics.confusion_matrix(y_train, BCmodel.predict(X_train)))
    fpr, tpr, thresholds = metrics.roc_curve(y_train, BCmodel.predict_proba(X_train)[:, 1], pos_label=1)
    score = metrics.roc_auc_score(y_train, BCmodel.predict_proba(X_train)[:, 1])
    fig = plt.figure(1)
    plt.plot(fpr, tpr, color='orange', lw=2, label="ROC curve(Area = %0.3f)" % score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title("Bayesian classifier ROC (Training Set)")
    plt.legend(loc = 'lower right')
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Postive Rate(TPR)')
    plt.savefig("Bayesian classifier ROC(Training Set).png")
    
    # (Testing Set) Calculate Bayesian Classification confusion Matrix、ROC curve and AUC score, and plot it
    print("Bayesian Classifier confusion Matrix (Testing Set) :\n", metrics.confusion_matrix(y_test, BCmodel.predict(X_test)))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, BCmodel.predict_proba(X_test)[:, 1], pos_label=1)
    score = metrics.roc_auc_score(y_test, BCmodel.predict_proba(X_test)[:, 1])
    fig2 = plt.figure(2)
    plt.plot(fpr, tpr, color='orange', lw=2, label="ROC curve(Area = %0.3f)" % score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title("Bayesian classifier ROC (Testing Set)")
    plt.legend(loc = 'lower right')
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Postive Rate(TPR)')
    plt.savefig("Bayesian classifier ROC(Testing Set).png")
    
    # (Training Set) Calculate Naive-Bayes classification confusion Matrix、ROC curve and AUC score, and plot it
    print("Naive-Bayes Classifier confusion Matrix (Training Set) :\n", metrics.confusion_matrix(y_train, NBmodel.predict(X_train)))
    fpr, tpr, thresholds = metrics.roc_curve(y_train, NBmodel.predict_proba(X_train)[:, 1], pos_label=1)
    score = metrics.roc_auc_score(y_train, NBmodel.predict_proba(X_train)[:, 1])
    fig3 = plt.figure(3)
    plt.plot(fpr, tpr, color='orange', lw=2, label="ROC curve(Area = %0.3f)" % score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title("Naive-Bayes classifier ROC (Training Set)")
    plt.legend(loc = 'lower right')
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Postive Rate(TPR)')
    plt.savefig("Naive-Bayes classifier ROC(Training Set).png")
    
    # (Testing Set) Calculate Naive-Bayes classification confusion Matrix、ROC curve and AUC score, and plot it
    print("Naive-Bayes Classifier confusion Matrix (Testing Set) :\n", metrics.confusion_matrix(y_test, NBmodel.predict(X_test)))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, NBmodel.predict_proba(X_test)[:, 1], pos_label=1)
    score = metrics.roc_auc_score(y_test, NBmodel.predict_proba(X_test)[:, 1])
    fig4 = plt.figure(4)
    plt.plot(fpr, tpr, color='orange', lw=2, label="ROC curve(Area = %0.3f)" % score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title("Naive-Bayes classifier ROC (Testing Set)")
    plt.legend(loc = 'lower right')
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Postive Rate(TPR)')
    plt.savefig("Naive-Bayes classifier ROC(Testing Set).png")
    
    
    plt.figure(figsize=(20, 20))
    plt.show()

    # linear classifier 
    LCmodel = linearClassifierGD(eta=0.000001, n_iterations = 100000)
    LCmodel.fit(X_train, y_train)
    # label data to [-1 1] for linear classficaition
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    print("Training Data Score-> Linear Classifier:", LCmodel.score(X_train, y_train))
    print("Testing Data Score-> Linear Classifier:", LCmodel.score(X_test, y_test))
    print("Average Score of Cross-Validation : Linear Classifier:", np.sum(cv.score(LCmodel, X_train, y_train, 3)) / 3)
    # Calculate linear classifier confusion Matrix
    print("linear Classifier confusion Matrix (Training Set) :\n", metrics.confusion_matrix(y_train, LCmodel.predict(X_train)))
    print("linear Classifier confusion Matrix (Testing Set) :\n", metrics.confusion_matrix(y_test, LCmodel.predict(X_test)))

if __name__ == '__main__':
    # Dataset1 - iris dataset
    print("----------1. Iris Dataset------------")
    iris = datasets.load_iris()  # load iris dataset
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state=42)  # Split dataset into trianing and testing randomly
    BCmodel = GaussianBC()
    NBmodel = GaussianNaiveBayes()
    BCmodel.fit(X_train, y_train)
    NBmodel.fit(X_train, y_train)
    cv = cross_validate()
    print("Training Data Score-> BC:", BCmodel.score(X_train, y_train), ", NB:", NBmodel.score(X_train, y_train))
    print("Testing Data Score-> BC:", BCmodel.score(X_test, y_test), ", NB:", NBmodel.score(X_test, y_test))
    print("Average Score of Cross-Validation : BC:", np.sum(cv.score(BCmodel, X_train, y_train, 3)) / 3, " NB:", np.sum(cv.score(NBmodel, X_train, y_train, 3)) / 3)

    # Dataset2 - Wheat-Seeds Dataset
    print("----------2. Wheat-Seeds Dataset------------")
    seeds=pd.read_csv('seeds.csv')
    train, test = train_test_split(seeds, test_size=0.2) # Split dataset into trianing and testing randomly

    # Data spliting(split features and ground-true label) and processing
    X_train = train.iloc[:, :7]  
    y_train = train.iloc[:, 7] - 1  # training data label (All value minus one, Because originally 1 ~ 3)
    X_test = test.iloc[:, :7]
    y_test = test.iloc[:, 7] - 1   # testing data label (All value minus one, Because originally 1 ~ 3)
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    BCmodel = GaussianBC()
    NBmodel = GaussianNaiveBayes()
    BCmodel.fit(X_train, y_train)
    NBmodel.fit(X_train, y_train)
    cv = cross_validate()
    print("Training Data Score-> BC:", BCmodel.score(X_train, y_train), ", NB:", NBmodel.score(X_train, y_train))
    print("Testing Data Score-> BC:", BCmodel.score(X_test, y_test), ", NB:", NBmodel.score(X_test, y_test))
    print("Average Score of Cross-Validation : BC:", np.sum(cv.score(BCmodel, X_train, y_train, 3)) / 3, " NB:", np.sum(cv.score(NBmodel, X_train, y_train, 3)) / 3)


    # Dataset3 - Breast Cancer Wisconsin (Diagnostic) DataSet
    print("----------3. Breast Cancer Wisconsin (Diagnostic) DataSet------------")
    breast_cancer = datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2, stratify=breast_cancer.target)
    binary_Classification_analysis(X_train, y_train, X_test, y_test)

    # Dataset4 - Pima Indians Diabetes Database
    print("----------4. Pima Indians Diabetes Database------------")
    diab = pd.read_csv('diabetes.csv')
    print(diab.isnull().sum()) # checking the data
    outcome = diab['Outcome']
    data = diab[diab.columns[:8]]
    train, test = train_test_split(diab, test_size=0.2, stratify=diab['Outcome']) #stratify the outcome
    X_train = train[train.columns[:8]]
    y_train = train['Outcome']
    X_test = test[test.columns[:8]]
    y_test = test['Outcome']
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    binary_Classification_analysis(X_train, y_train, X_test, y_test)


