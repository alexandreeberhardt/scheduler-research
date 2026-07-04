from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    top_k: tuple[int, ...] = (1, 3, 5),
) -> tuple[list[int], list[int], dict[int, list[bool]]]:
    """
    Parcourt le DataLoader et collecte :
        all_targets : liste des vrais labels
        all_preds : liste des prédictions top-1
        topk_correct : {k: liste des top-k prédictions} pour k > 1
    """
    model.eval()
    all_targets: list[int] = []
    all_top1: list[int] = []
    topk_correct: dict[int, list[bool]] = {k: [] for k in top_k if k > 1}

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        top1 = logits.argmax(dim=1)
        all_targets.extend(y.cpu().tolist())
        all_top1.extend(top1.cpu().tolist())

        for k in top_k:
            if k == 1:
                continue
            _, topk = logits.topk(k, dim=1)
            hit = (topk == y.unsqueeze(1)).any(dim=1)
            topk_correct[k].extend(hit.cpu().tolist())

    return all_targets, all_top1, topk_correct


def classification_report(
    targets: list[int],
    preds: list[int],
    id_to_token: dict[int, str],
    top_n: int = 50,
) -> str:
    """
    Genere un rapport de classification (precision / rappel / F1)
    pour les `top_n` classes les plus frequentes.
    """
    from collections import Counter, defaultdict

    class_counts = Counter(targets)
    top_classes = [cls for cls, _ in class_counts.most_common(top_n)]

    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    fn: dict[int, int] = defaultdict(int)

    for t, p in zip(targets, preds):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1

    lines = [
        f"{'Classe':<35} {'Support':>8}  {'Precision':>10}  {'Rappel':>8}  {'F1':>8}",
        "-" * 75,
    ]
    macro_p = macro_r = macro_f1 = 0.0
    n_classes = len(top_classes)

    for cls in top_classes:
        support = class_counts[cls]
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        rec = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        macro_p += prec
        macro_r += rec
        macro_f1 += f1
        name = id_to_token.get(cls, f"<id:{cls}>")
        lines.append(f"{name:<35} {support:>8}  {prec:>10.4f}  {rec:>8.4f}  {f1:>8.4f}")

    lines.append("-" * 75)
    lines.append(
        f"{'macro avg (top-' + str(top_n) + ' classes)':<35} "
        f"{'':>8}  {macro_p/n_classes:>10.4f}  {macro_r/n_classes:>8.4f}  "
        f"{macro_f1/n_classes:>8.4f}"
    )
    return "\n".join(lines)


def plot_confusion_matrix(
    targets: list[int],
    preds: list[int],
    id_to_token: dict[int, str],
    top_n: int,
    out_path: Path,
) -> None:
    """
    Trace et sauvegarde la matrice de confusion normalisee pour les `top_n`
    classes les plus frequentes dans le jeu de test.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib non disponible - confusion matrix ignoree")
        return

    from collections import Counter

    top_classes = [cls for cls, _ in Counter(targets).most_common(top_n)]
    cls_set = set(top_classes)
    idx = {cls: i for i, cls in enumerate(top_classes)}
    n = len(top_classes)

    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(targets, preds):
        if t in cls_set and p in cls_set:
            cm[idx[t], idx[p]] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    labels = [id_to_token.get(cls, f"<{cls}>") for cls in top_classes]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Precision normalisee")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predit")
    ax.set_ylabel("Reel")
    ax.set_title(f"Matrice de confusion - top {n} classes (normalisee)")

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            if val > 0.01:
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if val > 0.5 else "black",
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Matrice de confusion sauvegardee : {out_path}")
