#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from dpr_scale.utils.utils import ScriptEncoder
from dpr_scale.task.dpr_task import DenseRetrieverTask
from pytorch_lightning.strategies import DDPShardedStrategy, DDPStrategy


class DPRPairDistillTask(DenseRetrieverTask):
    def __init__(
        self,
        use_pair_scores: bool = True,
        pair_loss_alpha: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.use_pair_scores = use_pair_scores
        self.pair_loss_alpha = pair_loss_alpha

    def training_step(self, batch, batch_idx):
        """
        This receives queries, each with multiple contexts.
        """
        query_ids = batch["query_ids"]  # bs x tokens
        contexts_ids = batch["contexts_ids"]  # ctx_cnt x ctx_len
        target_scores = batch["scores"]  # bs x ctx_cnt
        pair_scores = batch["pair_scores"]
        query_repr, context_repr = self(query_ids, contexts_ids)  # bs

        queries_per_node = query_repr.size(0)
        contexts_per_node = context_repr.size(0)
        contexts_per_query = contexts_per_node // queries_per_node

        if self.in_batch_negatives:
            # gather all tensors for training w/ in_batch_negatives
            if isinstance(self.trainer.strategy, (DDPStrategy, DDPShardedStrategy)):
                query_to_send = query_repr.detach()
                context_to_send = context_repr.detach()
                # assumes all nodes have same number of contexts
                (
                    all_query_repr,
                    all_context_repr,
                    all_target_scores,
                ) = self.all_gather(
                    (query_to_send, context_to_send, target_scores)
                )

                all_query_list = []
                all_context_list = []

                world_size = all_query_repr.size(0)

                target_scores = torch.full(
                    size=(queries_per_node * world_size, contexts_per_node * world_size),
                    fill_value=-1e9,
                    device=all_target_scores.device
                )

                for i in range(world_size):
                    if i != self.global_rank:
                        all_query_list.append(all_query_repr[i])
                        all_context_list.append(all_context_repr[i])
                    else:
                        # to calculate grads for this node only
                        all_query_list.append(query_repr)
                        all_context_list.append(context_repr)
                    
                    for j in range(queries_per_node):
                        # fill in the scores tensor
                        target_scores[
                            i * queries_per_node + j,
                            i * contexts_per_node + j * contexts_per_query : i * contexts_per_node + (j + 1) * contexts_per_query
                        ] = all_target_scores[i][j]

                context_repr = torch.cat(all_context_list, dim=0)  # total_ctx x dim
                query_repr = torch.cat(all_query_list, dim=0)  # total_query x dim
        else:
            full_target_scores = torch.full(
                size=(queries_per_node, contexts_per_node),
                fill_value=-1e9,
                device=target_scores.device
            )
            for i in range(queries_per_node):
                full_target_scores[i, i * contexts_per_query : (i + 1) * contexts_per_query] = target_scores[i]
            target_scores = full_target_scores

        scores = self.sim_score(query_repr, context_repr, mask=None)  # total_query x total_ctx
        # temperature scaling
        scores /= self.softmax_temperature
        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
        log_target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)
        loss = self.loss(log_scores, log_target_scores)

        if self.use_pair_scores:
            scores_this_node = []
            if self.in_batch_negatives and isinstance(self.trainer.strategy, (DDPStrategy, DDPShardedStrategy)):
                for i in range(queries_per_node):
                    scores_this_node.append(
                        scores[
                            self.global_rank * queries_per_node + i,
                            self.global_rank * contexts_per_node + i * contexts_per_query : self.global_rank * contexts_per_node + (i + 1) * contexts_per_query
                        ]
                    )
                scores_this_node = torch.stack(scores_this_node, dim=0)
            else:
                for i in range(queries_per_node):
                    scores_this_node.append(scores[i, i * contexts_per_query : (i + 1) * contexts_per_query])
                scores_this_node = torch.stack(scores_this_node, dim=0)

            pair_loss = self._pair_distill_loss(scores_this_node, pair_scores)
            loss += pair_loss * self.pair_loss_alpha

        self.log("train_loss", loss, prog_bar=True)
        if self.use_pair_scores:
            self.log("train_pair_loss", pair_loss, prog_bar=True)
        return loss
    
    def _pair_distill_loss(self, scores, pair_scores):
        """
        Pairwise distillation loss function.
        :param scores: (batch_size, n_way) tensor of predicted scores.
        :param pair_scores: List[List[Tuple[int, float]]] of scores for each pair.
        :return: scalar loss.
        """

        # pair_scores[batch_idx][i] = [(j, score), ...] means that for the batch_idx-th query,
        # the i-th passage is more relevant than the j-th passage with a probability of 'score'.
        # Note that 0 <= len(pair_scores[batch_idx][i]) <= n_way - 1.

        # We first collect all the pairs.
        pairs = []
        for batch_idx, pair_score in enumerate(pair_scores):
            for i, passage_pairs in enumerate(pair_score):
                for j, score in passage_pairs:
                    pairs.append((batch_idx, i, j, math.exp(score)))

        if len(pairs) == 0:
            return torch.tensor(0.0, device=scores.device)

        # We then compute the loss.
        # The objective is to make softmax([scores[batch_idx][i], scores[batch_idx][j]])[0] = score
        # We use KL-divergence to measure the difference between the two distributions.
        kl_div = nn.KLDivLoss(reduction='batchmean', log_target=False)
        loss = 0.0
        for batch_idx, i, j, score in pairs:
            loss += kl_div(
                F.log_softmax(scores[batch_idx][[i, j]], dim=-1).unsqueeze(0),
                torch.tensor([score, 1 - score], device=scores.device).unsqueeze(0)
            )

        return loss / len(pairs)

    def _eval_step(self, batch, batch_idx):
        query_ids = batch["query_ids"]  # bs x tokens
        contexts_ids = batch["contexts_ids"]  # (bs x ctx_cnt, ctx_len)
        target_scores = batch["scores"]  # bs x ctx_cnt
        pair_scores = batch["pair_scores"]
        query_repr, contexts_repr = self(query_ids, contexts_ids)
        scores = self.sim_score(query_repr, contexts_repr, mask=None)
        scores /= self.softmax_temperature
        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)

        full_target_scores = torch.full(
            size=(query_repr.size(0), contexts_repr.size(0)),
            fill_value=-1e9,
            device=target_scores.device
        )
        contexts_per_query = contexts_repr.size(0) // query_repr.size(0)
        for i in range(query_repr.size(0)):
            full_target_scores[i, i * contexts_per_query : (i + 1) * contexts_per_query] = target_scores[i]
        target_scores = full_target_scores
        log_target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)
        loss = self.loss(log_scores, log_target_scores)

        if self.use_pair_scores:
            scores_for_pair = []
            for i in range(query_repr.size(0)):
                scores_for_pair.append(scores[i, i * contexts_per_query : (i + 1) * contexts_per_query])
            scores_for_pair = torch.stack(scores_for_pair, dim=0)
            pair_loss = self._pair_distill_loss(scores_for_pair, pair_scores)
            loss += pair_loss * self.pair_loss_alpha
        else:
            pair_loss = 0.0

        return (
            query_repr,
            contexts_repr,
            loss,
            pair_loss,
        )

    def _eval_epoch_end(self, outputs, log_prefix="valid"):
        total_ctx_count, total_count = 0, 0
        total_loss = 0
        total_pair_loss = 0
        if self.in_batch_eval:
            for query_repr, contexts_repr, loss, pair_loss in outputs:
                total_ctx_count += contexts_repr.size(0)
                total_count += query_repr.size(0)
                total_loss += loss
                total_pair_loss += pair_loss
            total_ctx_count = total_ctx_count / len(outputs)
            total_loss = total_loss / len(outputs)
            total_pair_loss = total_pair_loss / len(outputs)
        else:
            raise NotImplementedError

        metrics = {
            log_prefix + "_ctx_count": total_ctx_count,
            log_prefix + "_loss": total_loss,
            log_prefix + "_pair_loss": total_pair_loss,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
