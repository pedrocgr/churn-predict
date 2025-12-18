import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- THE PLOTTING FUNCTION ---
def plot_model_performance(model, X, y, title="Model Performance"):
    # Get probabilities
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = model.predict(X)
    
    y_pred = (y_proba >= 0.5).astype(int)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.suptitle(title, fontsize=16, y=1.05)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc_score = roc_auc_score(y, y_proba)
    axes[1].plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", color='darkorange')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    
    # 3. KS Plot
    class0 = y_proba[y == 0]
    class1 = y_proba[y == 1]
    ks_stat = ks_2samp(class1, class0).statistic
    
    sns.ecdfplot(class0, label='Class 0', ax=axes[2], color='blue')
    sns.ecdfplot(class1, label='Class 1', ax=axes[2], color='red')
    axes[2].set_title(f'KS Plot (KS = {ks_stat:.3f})')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()


