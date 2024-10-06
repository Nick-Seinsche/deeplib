"""Test the linear machine learning models."""

import numpy as np

import deeplib.from_scratch.linear_ml_models as models


def test_logistic_regression_on_exam_dataset(exam_dataset_train):
    """Test logistic regression on exam dataset."""
    M = models.LogisticRegression()
    Xtrain, ytrain = exam_dataset_train
    M.fit(Xtrain, ytrain)
    prd1 = M.predict(np.array([[1, 15, 7], [6, 95, 7]]))
    assert (prd1 == np.array([0, 1])).all()


def test_binary_logistic_regression_on_mnist_dataset(
    mnist_dataset_train, mnist_dataset_test
):
    """Test logistic regression on 0 recognition of mnist data."""
    M = models.LogisticRegression()
    Xtrain, ytrain = mnist_dataset_train
    Xtrain = Xtrain[:1000]
    ytrain = ytrain[:1000]
    ytrain_binary = (ytrain == 0).astype(int)
    M.fit(Xtrain, ytrain_binary)
    Xtest, ytest = mnist_dataset_test
    Xtest = Xtest[:1000]
    ytest = ytest[:1000]
    ytest_binary = (ytest == 0).astype(int)
    prd1 = M.predict(Xtest)
    assert np.mean(prd1 == ytest_binary) >= 0.95


def test_multinomial_logistic_regression_on_mnist_dataset(
    mnist_dataset_train, mnist_dataset_test
):
    """Test multinomial logistc regression on mnist dataset."""
    M = models.LogisticRegression()
    Xtrain, ytrain = mnist_dataset_train
    Xtrain = Xtrain[:1000]
    ytrain = ytrain[:1000]
    M.fit(Xtrain, ytrain)
    Xtest, ytest = mnist_dataset_test
    Xtest = Xtest[:1000]
    ytest = ytest[:1000]
    prd1 = M.predict(Xtest)
    assert np.mean(prd1 == ytest) >= 0.82
