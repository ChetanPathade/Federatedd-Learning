# client.py
import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.utils import shuffle
from utils import load_client_data

class GenomicClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = LogisticRegression(max_iter=100)
        self.X_train, self.X_test, self.y_train, self.y_test = load_client_data(client_id)
        self.train_membership_log = {}         # For prediction-based MIA
        self.per_sample_gradients = []         # For gradient-based MIA

    def get_parameters(self, config):
        try:
            return [self.model.coef_, self.model.intercept_]
        except AttributeError:
            # Return dummy values before training
            n_features = self.X_train.shape[1]
            return [np.zeros((1, n_features)), np.zeros(1)]

    def fit(self, parameters, config):
        if parameters:
            self.model.coef_ = parameters[0]
            self.model.intercept_ = parameters[1]

        # Log gradient-based info using SGDClassifier
        model = SGDClassifier(loss="log_loss", max_iter=1, learning_rate='constant', eta0=0.01, warm_start=True)
        X, y = shuffle(self.X_train.values, self.y_train.values)

        for i in range(len(X)):
            model.partial_fit(X[i:i+1], y[i:i+1], classes=np.array([0, 1]))
            grad = model.coef_.copy()
            self.per_sample_gradients.append({
                "gradient": grad.flatten(),
                "is_member": 1
            })
        # Train the main model
        self.model.fit(self.X_train, self.y_train)

        # Log prediction-based membership scores
        train_preds = self.model.predict_proba(self.X_train)[:, 1]
        for i, prob in enumerate(train_preds):
            self.train_membership_log[i] = (self.X_train.iloc[i].values, prob, 1)  # 1 = member

        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        if parameters:
            self.model.coef_ = parameters[0]
            self.model.intercept_ = parameters[1]

        # Prediction-based MIA (on non-members)
        preds = self.model.predict_proba(self.X_test)[:, 1]
        for i, prob in enumerate(preds):
            self.train_membership_log[len(self.train_membership_log)] = (self.X_test.iloc[i].values, prob, 0)  # 0 = non-member

        # Gradient-based MIA (on non-members)
        model = SGDClassifier(loss="log_loss", max_iter=1, learning_rate='constant', eta0=0.01, warm_start=True)
        model.partial_fit(self.X_train, self.y_train, classes=np.array([0, 1]))  # Init model

        for i in range(len(self.X_test)):
            x = self.X_test.iloc[i:i+1].values
            y = self.y_test.iloc[i]
            model.partial_fit(x, [y])
            grad = model.coef_.copy()
            self.per_sample_gradients.append({
                "gradient": grad.flatten(),
                "is_member": 0
            })

        loss = log_loss(self.y_test, preds)
        acc = accuracy_score(self.y_test, (preds > 0.5).astype(int))
        return loss, len(self.X_test), {"accuracy": acc}