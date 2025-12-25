# lggsn_model.py
# 简化版 LGGSN：只用几何特征做抓取好坏预测

import torch
import torch.nn as nn


class LGGSN(nn.Module):
    """
    Lightweight Grasp Geometry Scoring Network.

    目前只用几何 + 质量特征 (geom) 来预测抓取质量 label。
    预留了 query embedding 接口，方便以后接语言/类信息。

    Args:
        n_queries: 查询 id 的总数（现在用不到，先留接口）
        geom_dim:  几何特征维度（= feature_cols 的长度）
        query_dim: query embedding 维度（现在我们设成 0，相当于不用）
        hidden_dim: MLP 隐藏层维度
    """
    def __init__(
        self,
        n_queries: int,
        geom_dim: int = 12,
        query_dim: int = 0,
        hidden_dim: int = 40,
    ):
        super().__init__()
        self.use_query = query_dim > 0

        if self.use_query:
            self.query_emb = nn.Embedding(n_queries, query_dim)
        else:
            self.query_emb = None

        in_dim = geom_dim + (query_dim if self.use_query else 0)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, geom: torch.Tensor, query_id: torch.Tensor):
        """
        geom: [B, geom_dim]
        query_id: [B]，如果 use_query=False 会被忽略
        """
        x = geom
        if self.use_query:
            q = self.query_emb(query_id)  # [B, query_dim]
            x = torch.cat([geom, q], dim=-1)

        logit = self.mlp(x).squeeze(-1)  # [B]
        return logit

