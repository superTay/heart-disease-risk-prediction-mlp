import numpy as np
from sklearn.model_selection import KFold

# Importamos los datos ya limpios (X, y) desde el módulo de cleaning
from data_cleaning import X, y

print(f"Dataset shape -> X: {X.shape}, y: {y.shape}")

# 1) Definimos el número de divisiones (folds)
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 2) Recorremos los folds y mostramos las dimensiones de train/test
fold_id = 0
all_test_indices = []

for train_index, test_index in kf.split(X):
    fold_id += 1

    # Si X e y son DataFrame/Series de pandas, usamos .iloc
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"\nFold {fold_id}/{n_splits}")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_test  shape: {X_test.shape}, y_test  shape: {y_test.shape}")

    # Guardamos los índices de test para comprobar la cobertura total al final
    all_test_indices.extend(test_index)

# 3) Comprobación de que el dataset se ha dividido en 5 partes sin solaparse
all_test_indices = np.array(all_test_indices)

print("\n--- K-Fold split check ---")
print(f"Total samples in dataset:      {X.shape[0]}")
print(f"Total test indices seen:       {len(all_test_indices)}")
print(f"Unique test indices:           {len(np.unique(all_test_indices))}")

# Checks de integridad (profesional)
assert len(all_test_indices) == X.shape[0], \
    "Some samples were not included in any test fold."

assert len(np.unique(all_test_indices)) == X.shape[0], \
    "Some samples appear multiple times in test folds."

print("K-Fold integrity check passed ✅")