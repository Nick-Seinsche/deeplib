import numpy as np
import pickle as pkl


class Model:
    def save_model(self, path):
        with open(path, "wb") as f:
            pkl.dump(self, f)

    @staticmethod
    def load_model(path):
        with open(path, "rb") as f:
            return pkl.load(f)


class LinearRegression(Model):
    def __init__(self):
        self.w = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        # solve the Gaussian normal equation for ||Xw - y||^2
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return X @ self.w


class LogisticRegression(Model):
    def __init__(self, alpha=0):
        self.w = None
        self.X = None
        self.y = None
        self.aX = None
        self.alpha = alpha
        self.mode = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _cross_entropy_loss(self, step=0):
        weights = self.w + step
        z = self.aX @ weights

        if self.mode == "binary":
            s = self._sigmoid(z)
            loss = -np.mean(
                self.y * np.log(s + 1e-15) + (1 - self.y) * np.log(1 - s + 1e-15)
            )
            reg_term = self.alpha * np.dot(self.w, self.w)
            return loss + reg_term
        elif self.mode == "multinomial":
            s = self._softmax(z)
            loss = -np.mean(np.sum(self.y * np.log(s + 1e-15), axis=1))
            reg_term = self.alpha * np.sum(self.w * self.w)
            return loss + reg_term

    def _dx_cross_entropy_loss(self):
        z = self.aX @ self.w
        if self.mode == "binary":
            s = self._sigmoid(z)
            gradient = (
                self.aX.T @ (s - self.y)
            ) / self.y.size + 2 * self.alpha * self.w
            return gradient
        elif self.mode == "multinomial":
            s = self._softmax(z)
            gradient = (self.aX.T @ (s - self.y)) / self.y.shape[
                0
            ] + 2 * self.alpha * self.w
            return gradient

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.aX = np.hstack((np.ones((X.shape[0], 1)), X))

        if len(np.unique(y)) <= 2:
            self.mode = "binary"
            self.w = np.zeros(self.aX.shape[1])
        else:
            self.mode = "multinomial"
            K = len(np.unique(y))
            self.w = np.zeros((self.aX.shape[1], K))
            # One-hot encode y
            self.y = np.eye(K)[y]

        loss_before_step = self._cross_entropy_loss()
        loss_after_step = 0
        train_iterations = 0

        while (
            abs(loss_before_step - loss_after_step) > 1e-5 and train_iterations < 1000
        ):
            train_iterations += 1
            stepsize = 1

            gradient = self._dx_cross_entropy_loss()
            loss_before_step = self._cross_entropy_loss()

            # Line search to find the optimal step size
            while True:
                loss_after_step = self._cross_entropy_loss(step=-stepsize * gradient)

                if loss_after_step <= loss_before_step - 0.5 * stepsize * np.sum(
                    gradient * gradient
                ):
                    break

                stepsize *= 0.5

            print(f"Accepted stepsize: {stepsize}")
            self.w -= stepsize * gradient

        print(f"Fitting completed after {train_iterations} iterations")
        loss = self._cross_entropy_loss()
        print(f"Training loss is {loss}")
        return loss

    def predict(self, X):
        aX = np.hstack((np.ones((X.shape[0], 1)), X))
        z = aX @ self.w
        if self.mode == "binary":
            s = self._sigmoid(z)
            return (s >= 0.5).astype(int)
        elif self.mode == "multinomial":
            s = self._softmax(z)
            return np.argmax(s, axis=1)


if __name__ == "__main__":
    from pathlib import Path
    import idx2numpy

    pass
