"""End-of-run dashboard: side-by-side comparison of isolated vs hivemind."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from sim.metrics import RunMetrics
from sim.radionuclides import waste_classes


def render_comparison(
    results: dict[str, tuple[RunMetrics, object, list]],
    output_path: str | None = None,
    show: bool = True,
):
    """results: mode_name -> (metrics, coordinator, agents)."""
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    fig.suptitle(
        "Multi-agent radwaste classification: isolated vs hivemind",
        fontsize=14, fontweight="bold",
    )

    modes = list(results.keys())
    colors = {"isolated": "#dc8a47", "hivemind": "#4a9bd9"}

    # --- Headline bar charts: accuracy / dose / throughput ---
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_dose = fig.add_subplot(gs[0, 1])
    ax_tput = fig.add_subplot(gs[0, 2])

    accs = [results[m][0].accuracy() * 100 for m in modes]
    doses = [results[m][0].cumulative_dose_uSv for m in modes]
    tputs = [results[m][0].throughput_per_hour() for m in modes]
    bar_kw = dict(width=0.55)

    ax_acc.bar(modes, accs, color=[colors[m] for m in modes], **bar_kw)
    ax_acc.set_ylabel("Classification accuracy (%)")
    ax_acc.set_ylim(0, 100)
    for i, v in enumerate(accs):
        ax_acc.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax_acc.set_title("Accuracy — higher is better")

    ax_dose.bar(modes, doses, color=[colors[m] for m in modes], **bar_kw)
    ax_dose.set_ylabel("Cumulative worker dose (µSv)")
    ymax = max(doses) * 1.18 if doses else 1.0
    ax_dose.set_ylim(0, ymax)
    for i, v in enumerate(doses):
        ax_dose.text(i, v + ymax * 0.02, f"{v:.0f}", ha="center", fontweight="bold")
    ax_dose.set_title("Cumulative dose — lower is better")

    ax_tput.bar(modes, tputs, color=[colors[m] for m in modes], **bar_kw)
    ax_tput.set_ylabel("Throughput (items / sim hour)")
    ax_tput.set_ylim(0, max(tputs) * 1.18 if tputs else 1.0)
    for i, v in enumerate(tputs):
        ax_tput.text(i, v + max(tputs) * 0.02 if tputs else 0, f"{v:.1f}", ha="center", fontweight="bold")
    ax_tput.set_title("Throughput")

    # --- Accuracy-over-time curve ---
    ax_curve = fig.add_subplot(gs[0, 3])
    for m in modes:
        metrics: RunMetrics = results[m][0]
        if metrics.accuracy_history:
            ts, accs_t = zip(*metrics.accuracy_history)
            ax_curve.plot(np.asarray(ts) / 3600.0, np.asarray(accs_t) * 100,
                          label=m, color=colors[m], linewidth=2)
    ax_curve.set_xlabel("Sim hours")
    ax_curve.set_ylabel("Running accuracy (%)")
    ax_curve.set_title("Accuracy over time")
    ax_curve.legend(loc="lower right")
    ax_curve.set_ylim(0, 105)
    ax_curve.grid(True, alpha=0.3)

    # --- Confusion matrices ---
    for i, m in enumerate(modes):
        ax = fig.add_subplot(gs[1, i * 2:i * 2 + 2])
        metrics = results[m][0]
        classes, mat = metrics.confusion_matrix()
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix — {m}")
        for r in range(len(classes)):
            for c in range(len(classes)):
                ax.text(c, r, str(mat[r, c]), ha="center", va="center",
                        color="white" if mat[r, c] > mat.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04)

    # --- Coordinator learning timeline (hivemind only) ---
    ax_learn = fig.add_subplot(gs[2, :2])
    hivemind = results.get("hivemind")
    if hivemind is not None:
        coord = hivemind[1]
        if coord.retrain_events:
            versions = [e["version"] for e in coord.retrain_events]
            thresholds = [e.get("actinide_threshold") for e in coord.retrain_events]
            samples = [e["n_samples"] for e in coord.retrain_events]
            xs = versions
            ax_learn.plot(xs, [t if t is not None else 0 for t in thresholds],
                          "o-", color="#4a9bd9", linewidth=2, label="actinide threshold (NaI)")
            ax_learn2 = ax_learn.twinx()
            ax_learn2.plot(xs, samples, "s--", color="#8888aa", alpha=0.7, label="training samples")
            ax_learn.set_xlabel("Model version")
            ax_learn.set_ylabel("Actinide signature threshold")
            ax_learn2.set_ylabel("Training samples seen")
            ax_learn.set_title("Coordinator learning: threshold derived from HPGe rescrutiny labels")
            ax_learn.grid(True, alpha=0.3)
            ax_learn.legend(loc="lower right")
        else:
            ax_learn.text(0.5, 0.5, "No retrains in hivemind run",
                          ha="center", va="center", transform=ax_learn.transAxes)
            ax_learn.set_title("Coordinator learning")

    # --- Dose accumulation curves ---
    ax_dose_t = fig.add_subplot(gs[2, 2:])
    for m in modes:
        metrics = results[m][0]
        if metrics.dose_history:
            ts, ds = zip(*metrics.dose_history)
            ax_dose_t.plot(np.asarray(ts) / 3600.0, ds,
                           label=m, color=colors[m], linewidth=2)
    ax_dose_t.set_xlabel("Sim hours")
    ax_dose_t.set_ylabel("Cumulative dose (µSv)")
    ax_dose_t.set_title("Worker dose accumulation")
    ax_dose_t.legend()
    ax_dose_t.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=110, bbox_inches="tight")
    if show:
        plt.show()
    return fig
