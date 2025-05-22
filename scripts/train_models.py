import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from models import create_ann_model, create_cnn_model, create_lstm_model

# Load data
data = np.load('/kaggle/working/processed_data.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
X_train_cnn = data['X_train_cnn']
X_test_cnn = data['X_test_cnn']

# Compute class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train ANN
ann_model = create_ann_model(X_train.shape[1])
ann_history = ann_model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weight_dict,
    verbose=1
)
ann_model.save('/kaggle/working/ann_model.h5')

# Train CNN
cnn_model = create_cnn_model((X_train_cnn.shape[1], 1))
cnn_history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weight_dict,
    verbose=1
)
cnn_model.save('/kaggle/working/cnn_model.h5')

# Train LSTM
lstm_model = create_lstm_model((X_train_cnn.shape[1], 1))
lstm_history = lstm_model.fit(
    X_train_cnn, y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weight_dict,
    verbose=1
)
lstm_model.save('/kaggle/working/lstm_model.h5')
