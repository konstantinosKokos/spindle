from torch import Tensor
from torch.nn import Module, Linear, Dropout
from transformers import BertModel

from torch_geometric.nn import GlobalAttention

import pdb


class Encoder(Module):
    def __init__(self, core: str, sep_token_id: int, dropout_rate: float = 0.15):
        super().__init__()
        self.core = BertModel.from_pretrained(core)
        self.aggregator = GlobalAttention(gate_nn=Linear(768, 1))
        self.dropout = Dropout(dropout_rate)
        self.sep_token_id = sep_token_id

    def forward(self, token_ids: Tensor, attention_mask: Tensor, token_clusters: Tensor) -> Tensor:
        token_embeddings = self.core(token_ids, attention_mask=attention_mask)['last_hidden_state']
        # todo
        sparsity_mask = (attention_mask == 1).bitwise_and(token_ids != self.sep_token_id)
        token_embeddings = token_embeddings[sparsity_mask]
        token_embeddings = self.dropout(token_embeddings)
        return self.aggregator.forward(token_embeddings, token_clusters[sparsity_mask])