import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def plot_pred_vs_actual(y_true, y_pred, title="Predicted vs Actual", save_path=None):
    plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.7)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(lims, lims)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()

def plot_residuals(y_true, y_pred, title="Residuals vs Fitted", save_path=None):
    resid = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, resid, s=12, alpha=0.7)
    plt.axhline(0, ls="--")
    plt.xlabel("Fitted")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()

def report_mse(y_true, y_pred, label=""):
    mse = mean_squared_error(y_true, y_pred)
    print(f"{label} MSE: {mse:.4f}")
    return mse
