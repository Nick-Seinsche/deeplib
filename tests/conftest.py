"""Conftest."""

import pytest
import numpy as np
from pathlib import Path
import idx2numpy


@pytest.fixture
def exam_dataset_train():
    """Small exam sample training dataset."""
    # Features:
    # - hours studied
    # - score at the last exam out of 100
    # - hours of sleep
    Xtrain = np.array(
        [
            [0.5, 50, 6],
            [0.75, 55, 5],
            [1, 60, 7],
            [1.25, 65, 8],
            [1.5, 55, 6],
            [1.75, 60, 7],
            [2, 62, 6],
            [2.25, 70, 8],
            [2.5, 58, 6],
            [2.75, 65, 7],
            [3, 75, 7],
            [3.25, 70, 7],
            [3.5, 77, 6],
            [3.75, 74, 6],
            [4, 80, 8],
            [4.25, 85, 8],
            [4.5, 78, 7],
            [4.75, 82, 7],
            [5, 88, 8],
            [5.25, 89, 7],
            [5.5, 90, 8],
            [5.75, 92, 8],
            [6, 95, 9],
        ]
    )
    ytrain = np.array(
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
    )
    assert Xtrain.shape[0] == ytrain.shape[0]
    return (Xtrain, ytrain)


@pytest.fixture
def mnist_dataset_train():
    """Mnist Number training dataset."""
    path = Path("data/MNIST")
    train_images_file = "train-images.idx3-ubyte"
    train_labels_file = "train-labels.idx1-ubyte"

    train_images = idx2numpy.convert_from_file(str(path / train_images_file)).reshape(
        -1, 28 * 28
    )
    train_labels = idx2numpy.convert_from_file(str(path / train_labels_file))
    return (train_images, train_labels)


@pytest.fixture
def mnist_dataset_test():
    """Mnist Number test dataset."""
    path = Path("data/MNIST")
    test_images_file = "t10k-images.idx3-ubyte"
    test_labels_file = "t10k-labels.idx1-ubyte"

    test_images = idx2numpy.convert_from_file(str(path / test_images_file)).reshape(
        -1, 28 * 28
    )
    test_labels = idx2numpy.convert_from_file(str(path / test_labels_file))
    return (test_images, test_labels)
