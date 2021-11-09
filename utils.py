from typing import List

import torch
import torchmetrics

from collie.interactions import Interactions, ExplicitInteractions, InteractionsDataLoader
from collie.metrics import evaluate_in_batches
from collie.model import CollieTrainer, MatrixFactorizationModel, MultiStagePipeline

from metrics import mapk, mrr, auc


def train_multi_stage_model(
        model: MultiStagePipeline, trainer: CollieTrainer, val_interactions: Interactions, n_stages: int = 2,
        add_epochs: int = 5) -> MultiStagePipeline:
    for stage_index in range(n_stages):
        print(f"Stage: {stage_index}")
        if stage_index > 0:
            model.advance_stage()
            trainer.max_epochs += add_epochs

        trainer.fit(model)
        model.eval()  # set model to inference mode
        mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, model)

        print(f"MAP@10 Score: {mapk_score}")
        print(f"MRR Score:    {mrr_score}")
        print(f"AUC Score:    {auc_score}")
        print("========================================================================================")

    return model


def evaluate_explicit(
        model: MatrixFactorizationModel, explicit_interactions: ExplicitInteractions,
        metrics_list: List[torchmetrics.metric.Metric]) -> List[torch.Tensor]:
    test_loader = InteractionsDataLoader(interactions=explicit_interactions)
    for batch in test_loader:
        users, items, ratings = batch
        users = users.to(torch.int64)
        items = items.to(torch.int64)
        ratings = ratings.cpu()
        preds = model(users, items)

        for metric in metrics_list:
            metric(preds.cpu(), ratings)

    return [metric.compute() for metric in metrics_list]
