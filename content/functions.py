from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def plot_roc_curves(y_true, prob_dict, metrics_dict, title="ROC Curves on Test Set", save_path=None):
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle="--")

    for name, y_prob in prob_dict.items():
        y_prob = np.asarray(y_prob).ravel()
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        auc = metrics_dict[name].get("AUC", float("nan"))
        ci = metrics_dict[name].get("AUC_CI95", (float("nan"), float("nan")))

        label = f"{name} (AUC = {auc:.4f} [{ci[0]:.4f}--{ci[1]:.4f}])"
        plt.plot(fpr, tpr, label=label)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_calibration_curves_paper(y_true, prob_dict, models_to_plot,
                                  n_bins=5, save_path=None):
    plt.figure(figsize=(7, 7))

    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")

    for name in models_to_plot:
        if name not in prob_dict:
            continue

        y_prob = np.asarray(prob_dict[name]).ravel()

        frac_pos, mean_pred = calibration_curve(
            y_true,
            y_prob,
            n_bins=n_bins,
            strategy="uniform"
        )

        plt.plot(mean_pred, frac_pos, marker="o", label=name)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Calibration Curves on Test Set")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_curve_analysis(
    y_true,
    prob_dict,
    thresholds=None,
    title="Decision Curve Analysis",
    save_path=None,
    xlim=(0.01, 0.80),
    ylim=None
):
    """
    Plot Decision Curve Analysis (DCA).
    Show net benefit vs threshold.
    """

    # Convert to numpy array
    y_true = np.asarray(y_true).ravel()
    n = len(y_true)

    if n == 0:
        return

    # Set threshold range
    if thresholds is None:
        thresholds = np.linspace(xlim[0], xlim[1], 200)

    thresholds = np.asarray(thresholds)
    thresholds = thresholds[(thresholds > 0) & (thresholds < 1)]

    # Event rate
    prevalence = y_true.mean()

    plt.figure(figsize=(8, 6))

    all_net_benefits = []

    # Plot each model
    for name, y_prob in prob_dict.items():

        y_prob = np.asarray(y_prob).ravel()

        # Skip if length is wrong
        if len(y_prob) != n:
            continue

        net_benefits = []

        # Compute net benefit for each threshold
        for t in thresholds:

            # Weight for false positives
            w = t / (1 - t)

            # Binary prediction
            y_pred = (y_prob >= t).astype(int)

            # True positives
            tp = np.sum((y_true == 1) & (y_pred == 1))

            # False positives
            fp = np.sum((y_true == 0) & (y_pred == 1))

            # Net benefit formula
            nb = (tp / n) - (fp / n) * w

            net_benefits.append(nb)

        net_benefits = np.array(net_benefits)

        all_net_benefits.append(net_benefits)

        plt.plot(thresholds, net_benefits, label=name, linewidth=1.8)

    # Treat all line
    nb_treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    plt.plot(thresholds, nb_treat_all, "k--", label="Treat all", linewidth=1.5)

    # Treat none line
    plt.axhline(0, color="gray", linestyle=":", label="Treat none")

    # Set x range
    plt.xlim(xlim[0], xlim[1])

    # Set y range automatically
    if ylim is None and len(all_net_benefits) > 0:

        all_nb = np.vstack(all_net_benefits)

        low = np.percentile(all_nb, 2)
        high = np.percentile(all_nb, 98)

        pad = 0.1 * (high - low + 1e-6)

        plt.ylim(low - pad, high + pad)

    elif ylim is not None:
        plt.ylim(ylim)

    # Labels and title
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(title)

    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if needed
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()



def compute_threshold_metrics(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)

    results = []

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) > 0 else 0
        )

        youden = sensitivity + specificity - 1

        results.append({
            "threshold": round(t, 4),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "accuracy": accuracy,
            "youden": youden,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        })
    df = pd.DataFrame(results)
    return df