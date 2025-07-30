import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot


def plot_roc(ax, y_true, y_pred):
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_pred[sorted_indices]

    tpr = []
    fpr = []
    thresholds = np.linspace(0, 1, 100)

    for threshold in thresholds:
        y_pred = (y_prob_sorted >= threshold).astype(int)  # noqa

        TP = np.sum((y_pred == 1) & (y_true_sorted == 1))
        FP = np.sum((y_pred == 1) & (y_true_sorted == 0))
        FN = np.sum((y_pred == 0) & (y_true_sorted == 1))
        TN = np.sum((y_pred == 0) & (y_true_sorted == 0))

        tpr.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        fpr.append(FP / (FP + TN) if (FP + TN) > 0 else 0)

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    auc = np.trapezoid(tpr[np.argsort(fpr)], fpr[np.argsort(fpr)])

    ax.plot(fpr, tpr, label=f"ROC curve (AUC {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True)


def plot_training(logs, y_true, y_pred, directory, overrides):
    data = pd.DataFrame(logs)

    data.to_pickle(os.path.join(directory, "data.pkl"))

    plots = [(data, "all"), (data[-10:], "last")]

    for df, suffix in plots:
        file = os.path.join(directory, f"training_{suffix}")

        epoch = logs[-1]["epoch"]
        y = df["step"]

        fig, axes = pyplot.subplots(nrows=4, ncols=4, figsize=(24, 12))

        axes = axes.T

        graph_loss = df[["loss_graph", "loss_graph_pos", "loss_graph_neg"]]
        x_ents = df[["x_ent_train", "x_ent_val"]]

        grad_norm = df[["grad_norm"]]
        param_norm = df[["norm_cnn", "norm_gnn", "norm_mlp"]]

        metrics = df[["acc", "precision", "recall", "f1"]]

        lr = df[["lr"]]

        corrects = df[
            [
                "train_correct_pred",
                "train_match_pred",
                "val_correct_pred",
                "val_match_pred",
            ]
        ]

        durations = df[
            [
                "duration_epoch",
                "duration_train",
                "duration_eval",
                "duration_plot",
            ]
        ]
        buffer_sps = df[["buffer_sps_uni", "buffer_sps_hard"]]
        buffer_size = df[["buffer_size_pos", "buffer_size_neg"]]
        buffer_gpi = df[["buffer_gpi"]]
        pos_frac = df[["pos_frac_train", "pos_frac_true"]]
        logits = df[["logits"]]

        lens = df[["len_train", "len_val"]]

        top_k = df[["top_k_train", "top_k_val"]]

        system = df[
            [
                "gpu_util_mean",
                "gpu_util_max",
                "cpu_util_mean",
                "cpu_util_max",
                "gpu_mem_mean",
                "gpu_mem_max",
                "cpu_mem_mean",
                "cpu_mem_max",
            ]
        ]

        plots = [
            (graph_loss, 0, 0, None),
            (metrics, 0, 1, None),
            (logits, 0, 2, None),
            (pos_frac, 0, 3, None),
            (x_ents, 1, 0, None),
            (lens, 1, 1, None),
            (top_k, 1, 2, None),
            (corrects, 1, 3, None),
            (grad_norm, 2, 0, None),
            (buffer_sps, 2, 1, None),
            (buffer_gpi, 2, 2, None),
            (lr, 2, 3, None),
            (param_norm, 3, 0, None),
            (durations, 3, 1, None),
            (system, 3, 2, None),
            (buffer_size, 3, 3, None),
        ]

        for i, (data, row, col, clip) in enumerate(plots):
            ax = axes[row, col]
            if data is None:
                ax.axis("off")
                continue

            if isinstance(data, str) and data == "roc":
                plot_roc(ax, y_true, y_pred)
                continue
            label = [f"{k} ({v})" for k, v in data.round(4).iloc[-1].to_dict().items()]
            ax.plot(y, data, label=label)
            ax.legend(loc="upper left")
            ax.grid()
            if clip is not None:
                ax.set_ylim(*clip)

        groups = {}
        for o in overrides:
            key, value = o.split("=")
            key = key.split("@")[-1]
            if "." in key:
                parts = key.split(".")
                group = ".".join(parts[:-1])
                key = parts[-1]

                groups.setdefault(group, [])
                groups[group].append(f"{key}={value}")
            else:
                groups.setdefault(key, [])
                groups[key].append(f"{value}")

        n = 8
        rows = []
        for k, v in groups.items():
            for i in range(0, len(v), n):
                rows.append((k, v[i : i + n]))

        rows = [f"{k}: " + ", ".join(v) for k, v in rows]

        title = "\n".join(rows)

        date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

        fig.suptitle(f"epoch {epoch} {date}\n{title}")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - len(groups) * 0.01))
        fig.savefig(file)
        pyplot.close(fig)
