from typing import Any, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
import torch
from torchmetrics.functional import auroc


def _get_labels(
        targets: csr_matrix, user_ids: Tuple[np.array, torch.tensor], preds: Tuple[np.array, torch.tensor],
        device: str) -> torch.tensor:
    return torch.tensor(
        (targets[user_ids[:, None], np.array(preds.detach().cpu())] > 0).astype('double').toarray(),
        requires_grad=False, device=device,
    )


def mapk(
        targets: csr_matrix, user_ids: Tuple[np.array, torch.tensor], preds: Tuple[np.array, torch.tensor],
        k: int = 10) -> float:
    device = preds.device
    n_users = preds.shape[0]
    predicted_items = preds.topk(k, dim=1).indices
    topk_labeled = _get_labels(targets, user_ids, predicted_items, device)
    accuracy = topk_labeled.int()

    weights = (
        1.0 / torch.arange(start=1, end=k+1, dtype=torch.float64, requires_grad=False, device=device)
    ).repeat(n_users, 1)

    denominator = torch.min(
        torch.tensor(k, device=device, dtype=torch.int).repeat(len(user_ids)),
        torch.tensor(targets[user_ids].getnnz(axis=1), device=device)
    )

    res = ((accuracy * accuracy.cumsum(axis=1) * weights).sum(axis=1)) / denominator
    res[torch.isnan(res)] = 0
    return res.mean().item()


def mrr(
        targets: csr_matrix, user_ids: Tuple[np.array, torch.tensor], preds: Tuple[np.array, torch.tensor],
        k: Optional[int] = None) -> float:

    predicted_items = preds.topk(preds.shape[1], dim=1).indices
    labeled = _get_labels(targets, user_ids, predicted_items, device=preds.device)

    position_weight = 1.0/(
        torch.arange(1, targets.shape[1] + 1, device=preds.device)
        .repeat(len(user_ids), 1)
        .float()
    )
    labeled_weighted = (labeled.float() * position_weight)
    highest_score, rank = labeled_weighted.topk(k=1)
    reciprocal_rank = 1.0/(rank.float() + 1)
    reciprocal_rank[highest_score == 0] = 0
    return reciprocal_rank.mean().item()


def auc(
        targets: csr_matrix, user_ids: Tuple[np.array, torch.tensor], preds: Tuple[np.array, torch.tensor],
        k: Optional[int] = None) -> float:
    agg = 0
    for i, user_id in enumerate(user_ids):
        target_tensor = torch.tensor(
            targets[user_id].toarray(),
            device=preds.device,
            dtype=torch.long
        ).view(-1)
        agg += auroc(torch.sigmoid(preds[i, :]), target=target_tensor, pos_label=1)
    return (agg/len(user_ids)).item()
