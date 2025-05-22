import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score

# Load data
data = np.load('/kaggle/working/processed_data.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Compute class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model variants
def create_ann_base(input_shape):
    return Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

def create_ann_no_dropout(input_shape):
    return Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

def create_ann_fewer_layers(input_shape):
    return Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(1, activation='sigmoid')
    ])

def create_ann_mid_layer_removed(input_shape):
    return Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

ablation_models = {
    "Base ANN": create_ann_base,
    "No Dropout": create_ann_no_dropout,
    "Fewer Layers": create_ann_fewer_layers,
    "Mid Layer Removed": create_ann_mid_layer_removed
}

ablation_f1_scores = {}

for name, model_fn in ablation_models.items():
    print(f"Training: {name}")
    model = model_fn(X_train.shape[1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        class_weight=class_weight_dict,
        verbose=0
    )
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    ablation_f1_scores[name] = f1_score(y_test, y_pred)

# Save results
pd.DataFrame.from_dict(ablation_f1_scores, orient='index', columns=["F1 Score"]).to_excel('/kaggle/working/ablation_f1_scores.xlsx')

# Plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid", font_scale=1.2)
sorted_scores = dict(sorted(ablation_f1_scores.items(), key=lambda x: x[1], reverse=True))
labels = list(sorted_scores.keys())
scores = list(sorted_scores.values())
ax = sns.barplot(x=labels, y=scores, palette='deep')
for i, v in enumerate(scores):
    ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=11, fontweight='bold')
plt.title("Ablation Study on ANN Architecture")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig('/kaggle/working/ablation_ann_study_improved.svg', format='svg', dpi=300)
plt.show()
