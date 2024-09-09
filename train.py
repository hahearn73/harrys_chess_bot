import json
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

X_vectors = []
Y_vectors = []

with open('dataset/vectorized_10_000.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        x = np.array(data['vectors'])
        X_vectors.append(x)
        Y_vectors.append(np.array([data['eval']]))

X_vectors = np.array(X_vectors)
Y_vectors = np.array(Y_vectors)
print("Input features shape:", X_vectors.shape)
print("Target values shape:", Y_vectors.shape)
# scaler_X = MinMaxScaler(feature_range=(0, 1))
# X_vectors = scaler_X.fit_transform(X_vectors)
scaler_Y = MinMaxScaler(feature_range=(0, 1))
Y_vectors = scaler_Y.fit_transform(Y_vectors)

param_grid = {
    'hidden_layer_sizes': [(4096), (16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16), (128, 128, 128, 128, 128)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [10, 100, 200, 500, 1000]
}
grid_search = GridSearchCV(MLPRegressor(max_iter=10), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=3)
grid_search.fit(X_vectors, Y_vectors.ravel())
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", -grid_search.best_score_)
