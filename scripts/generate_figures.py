"""Generate publication-quality evaluation figures from verified v1 and v3 SRL results."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def plot_v1_vs_v3():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)

    # Panel 1: English PropBank-EWT
    en_labels = ["Rule-Based", "CRF", "Neural\n(Mono)", "Neural\n(Polyglot EN+ES)"]
    en_v1 = [0.3397, 0.5453, 0.7343, 0.7343]
    en_v3 = [0.4877, 0.7185, 0.8168, 0.8220]
    x = np.arange(len(en_labels))
    w = 0.35

    ax = axes[0]
    b1 = ax.bar(x - w / 2, en_v1, w, label="v1 Baseline (Original Paper)", color="#94a3b8", edgecolor="#475569")
    b2 = ax.bar(x + w / 2, en_v3, w, label="v3 Upgraded Pipeline", color="#2563eb", edgecolor="#1e3a8a")
    ax.set_title("English — PropBank-EWT (Full Test, n=4,795)")
    ax.set_ylabel("CoNLL-2005 Span F1")
    ax.set_xticks(x)
    ax.set_xticklabels(en_labels)
    ax.set_ylim(0, 0.96)
    ax.legend(loc="upper left", frameon=True)

    for rect, val in zip(b1, en_v1):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#475569")
    for rect, v_old, v_new in zip(b2, en_v1, en_v3):
        delta = (v_new - v_old) * 100
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v_new + 0.012,
            f"{v_new:.3f}\n(+{delta:.1f})",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#1e3a8a",
        )

    # Panel 2: Spanish PropBank-ES (AnCora)
    es_labels = ["Rule-Based", "Zero-Shot\nEN→ES", "Neural\n(Mono)", "Neural\n(Polyglot EN+ES)"]
    es_v1 = [0.1438, 0.0, 0.4594, 0.4594]
    es_v3 = [0.2285, 0.3335, 0.5340, 0.5419]
    x2 = np.arange(len(es_labels))

    ax2 = axes[1]
    b1_es = ax2.bar(x2 - w / 2, es_v1, w, label="v1 Baseline (Original Paper)", color="#94a3b8", edgecolor="#475569")
    b2_es = ax2.bar(x2 + w / 2, es_v3, w, label="v3 Upgraded Pipeline", color="#ea580c", edgecolor="#9a3412")
    # Reference dashed line for v1 supervised ES CRF (0.2952)
    ax2.axhline(0.2952, color="#64748b", linestyle=":", linewidth=1.3, label="v1 Supervised ES CRF (0.295)")
    ax2.set_title("Spanish — PropBank-ES / AnCora (Full Test, n=3,702)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(es_labels)
    ax2.legend(loc="upper left", frameon=True)

    for i, (rect, val) in enumerate(zip(b1_es, es_v1)):
        if val > 0:
            ax2.text(rect.get_x() + rect.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#475569")
        else:
            ax2.text(rect.get_x() + rect.get_width() / 2, 0.015, "N/A", ha="center", va="bottom", fontsize=8.5, color="#64748b")

    for i, (rect, v_old, v_new) in enumerate(zip(b2_es, es_v1, es_v3)):
        txt = f"{v_new:.3f}\n(0 ES data)" if i == 1 else f"{v_new:.3f}\n(+{(v_new - v_old)*100:.1f})"
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            v_new + 0.012,
            txt,
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#9a3412",
        )

    plt.suptitle("SRL Progression: v1 Paper Baseline vs. v3 Upgraded Pipeline", fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "v1_vs_v3_f1.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved", out)


def plot_encoder_vs_llm():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # Left: Bar chart F1 on 200-instance stratified slice (100 EN + 100 ES)
    models = [
        "Span-Biaffine\nXLM-R (Full)",
        "Span-Biaffine\nXLM-R (500-shot)",
        "Qwen2.5-3B\n(Zero-Shot)",
        "Qwen2.5-3B\n(5-Shot)",
        "Qwen2.5-3B\n(QLoRA 500)",
    ]
    en_f1 = [0.7824, 0.2586, 0.0878, 0.1383, 0.0354]
    es_f1 = [0.5916, 0.1935, 0.1013, 0.0694, 0.0347]
    x = np.arange(len(models))
    w = 0.36

    ax = axes[0]
    r1 = ax.bar(x - w / 2, en_f1, w, label="English Slice (n=100)", color="#2563eb")
    r2 = ax.bar(x + w / 2, es_f1, w, label="Spanish Slice (n=100)", color="#ea580c")
    ax.set_title("Span F1 on Stratified 200-Instance Slice")
    ax.set_ylabel("CoNLL-2005 Span F1")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.8)
    ax.set_ylim(0, 0.90)
    ax.legend(loc="upper right", frameon=True)
    for rect, v in zip(r1, en_f1):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.2, color="#1e3a8a", fontweight="bold")
    for rect, v in zip(r2, es_f1):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.2, color="#9a3412", fontweight="bold")

    # Right: Accuracy vs Latency (Log Scale) Pareto plot
    ax2 = axes[1]
    points = [
        ("XLM-R Full (EN)", 9.0, 0.7824, "#2563eb", "o", (12, -3)),
        ("XLM-R Full (ES)", 12.6, 0.5916, "#ea580c", "o", (12, -3)),
        ("XLM-R 500-shot (EN)", 8.3, 0.2586, "#2563eb", "s", (12, 4)),
        ("XLM-R 500-shot (ES)", 11.5, 0.1935, "#ea580c", "s", (12, -8)),
        ("Qwen-3B 0-shot (EN, 92% JSON)", 1909.4, 0.0878, "#16a34a", "^", (-175, 18)),
        ("Qwen-3B 0-shot (ES, 92% JSON)", 2230.3, 0.1013, "#15803d", "^", (-175, 42)),
        ("Qwen-3B 5-shot (EN, 81% JSON)", 5782.4, 0.1383, "#9333ea", "D", (-175, 48)),
        ("Qwen-3B QLoRA (EN, 68% JSON)", 4786.8, 0.0354, "#dc2626", "X", (-215, 2)),
    ]
    for name, lat, f1, col, mk, offset in points:
        ax2.scatter(lat, f1, color=col, marker=mk, s=95, zorder=4, edgecolors="black", linewidths=0.6)
        arrow = dict(arrowstyle="-", color=col, lw=0.8, alpha=0.65) if lat > 1000 else None
        ax2.annotate(name, (lat, f1), xytext=offset, textcoords="offset points", fontsize=8.0, color=col, fontweight="bold", arrowprops=arrow)

    ax2.set_xscale("log")
    ax2.set_xlim(4, 15000)
    ax2.set_ylim(-0.02, 0.88)
    ax2.set_xlabel("Inference Latency (ms / instance, log scale)")
    ax2.set_ylabel("CoNLL-2005 Span F1")
    ax2.set_title("Accuracy vs. Latency Trade-Off (278M Encoder vs. 3B LLM)")
    ax2.axvspan(5, 25, color="#dbeafe", alpha=0.35, label="Encoder Regime (~10 ms)")
    ax2.axvspan(1200, 10000, color="#fee2e2", alpha=0.35, label="3B LLM Regime (200x-550x slower)")
    ax2.legend(loc="center right", fontsize=8.5, frameon=True)

    plt.suptitle("Specialized 278M Span-Biaffine Encoder vs. 3B Generative LLM (Qwen2.5-3B-Instruct)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "encoder_vs_llm_tradeoff.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved", out)


def plot_error_analysis():
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    systems = [
        "EN Rule-Based v2",
        "EN CRF v2 (L2-SGD)",
        "EN Span-Biaffine Polyglot",
        "ES Rule-Based v2",
        "ES Span-Biaffine Polyglot",
    ]
    exact = np.array([4190, 6662, 7523, 1705, 3636])
    boundary = np.array([194, 330, 215, 1698, 806])
    role_conf = np.array([2630, 1186, 703, 963, 197])
    missed = np.array([2115, 897, 810, 2794, 3398])
    spurious = np.array([625, 767, 440, 1489, 590])

    totals = exact + boundary + role_conf + missed + spurious
    # Normalize to percentages for clean cross-dataset comparison
    p_ex = 100 * exact / totals
    p_bd = 100 * boundary / totals
    p_rc = 100 * role_conf / totals
    p_ms = 100 * missed / totals
    p_sp = 100 * spurious / totals

    y = np.arange(len(systems))[::-1]
    h = 0.56
    ax.barh(y, p_ex, h, label="Exact Match (TP)", color="#16a34a")
    ax.barh(y, p_bd, h, left=p_ex, label="Boundary Only", color="#eab308")
    ax.barh(y, p_rc, h, left=p_ex + p_bd, label="Role Confusion", color="#f97316")
    ax.barh(y, p_ms, h, left=p_ex + p_bd + p_rc, label="Missed Span (FN)", color="#dc2626")
    ax.barh(y, p_sp, h, left=p_ex + p_bd + p_rc + p_ms, label="Spurious Span (FP)", color="#7c3aed")

    for yi, ex_v, bd_v, rc_v, ms_v, sp_v in zip(y, p_ex, p_bd, p_rc, p_ms, p_sp):
        if ex_v >= 8:
            ax.text(ex_v / 2, yi, f"{ex_v:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if bd_v >= 7:
            ax.text(ex_v + bd_v / 2, yi, f"{bd_v:.0f}%", ha="center", va="center", color="#1e293b", fontweight="bold", fontsize=8.5)
        if rc_v >= 7:
            ax.text(ex_v + bd_v + rc_v / 2, yi, f"{rc_v:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
        if ms_v >= 7:
            ax.text(ex_v + bd_v + rc_v + ms_v / 2, yi, f"{ms_v:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
        if sp_v >= 7:
            ax.text(ex_v + bd_v + rc_v + ms_v + sp_v / 2, yi, f"{sp_v:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(systems, fontweight="bold")
    ax.set_xlabel("Share of All Predicted + Gold Span Events (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Span-Level Error Decomposition Across Paradigms & Languages")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=True, fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "error_analysis_breakdown.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved", out)


if __name__ == "__main__":
    plot_v1_vs_v3()
    plot_encoder_vs_llm()
    plot_error_analysis()
