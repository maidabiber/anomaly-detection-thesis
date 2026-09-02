import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


def fit_isolation_forest(X_healthy, random_state=42):
    model = IsolationForest(random_state=random_state)
    model.fit(X_healthy)
    return model


def score_isolation_forest(model, X):
    return -model.score_samples(X)


def fit_one_class_svm(X_healthy, nu=0.01, kernel="rbf", gamma="scale"):
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X_healthy)
    return model


def score_one_class_svm(model, X):
    return -model.decision_function(X)


def fit_lof(X_healthy, n_neighbors=20):
    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    model.fit(X_healthy)
    return model


def score_lof(model, X):
    return -model.score_samples(X)


def fit_pca(X_healthy, n_components=0.95):
    model = PCA(n_components=n_components)
    model.fit(X_healthy)
    return model


def score_pca(model, X):
    X_transformed = model.transform(X)
    X_reconstructed = model.inverse_transform(X_transformed)
    return np.sum((np.asarray(X) - X_reconstructed) ** 2, axis=1)