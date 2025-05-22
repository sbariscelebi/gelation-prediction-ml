import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load test and train data
data = np.load('/kaggle/working/processed_data.npz')
X_train = data['X_train']
X_test = data['X_test']

# Load model
from tensorflow.keras.models import load_model
ann_model = load_model('/kaggle/working/ann_model.h5')

# Select background and test samples
X_background = X_train[np.random.choice(X_train.shape[0], size=100, replace=False)]
X_test_sample = X_test[:50]
X_test_instance = X_test[0:1]

# SHAP Kernel Explainer
explainer = shap.KernelExplainer(ann_model.predict, X_background)
shap_values_summary = explainer.shap_values(X_test_sample)
shap_values_instance = explainer.shap_values(X_test_instance)

# Use feature names from dataset
import pandas as pd
df = pd.read_excel('/kaggle/input/mutluhocamingilteredata/data.xlsx')
feature_names = df.drop('Gelation', axis=1).columns
X_test_sample_df = pd.DataFrame(X_test_sample, columns=feature_names)

# Global SHAP summary plot
plt.figure()
shap.summary_plot(shap_values_summary[0], X_test_sample_df, plot_type="dot", show=False)
plt.title("SHAP Feature Importance (ANN)")
plt.tight_layout()
plt.savefig('/kaggle/working/shap_summary_dot_ann.svg', format='svg', dpi=300)
plt.show()
