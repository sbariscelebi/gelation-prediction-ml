import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_excel('/kaggle/input/mutluhocamingilteredata/data.xlsx')

# Outlier removal using Z-score
numerical_cols = df.select_dtypes(include=[np.number]).columns.drop('Gelation')
z_scores = np.abs(zscore(df[numerical_cols]))
df = df[(z_scores < 3).all(axis=1)].reset_index(drop=True)

# Train-test split
X = df.drop(['Gelation'], axis=1)
y = df['Gelation']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for CNN/LSTM
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Save processed data
np.savez('/kaggle/working/processed_data.npz',
         X_train=X_train_scaled, X_test=X_test_scaled,
         y_train=y_train, y_test=y_test,
         X_train_cnn=X_train_cnn, X_test_cnn=X_test_cnn)
