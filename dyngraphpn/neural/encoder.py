
from torch import Tensor
from torch.nn import Module, Linear
from transformers import BertModel, RobertaModel, CamembertModel

from torch_geometric.nn.aggr import AttentionalAggregation as GlobalAttention
from .utils import RMSNorm


class Encoder(Module):
    def __init__(self,
                 config_or_name: dict | str,
                 bert_type: str,
                 sep_token_id: int,
                 bottleneck_dim: int = 768):
        super().__init__()
        cls = {'bert': BertModel, 'roberta': RobertaModel, 'camembert': CamembertModel}[bert_type]
        self.core = cls.from_pretrained(config_or_name) if isinstance(config_or_name, str) else cls._from_config(config_or_name)
        self.aggregator = GlobalAttention(gate_nn=Linear(self.core.config.hidden_size, 1),
                                          nn=Linear(self.core.config.hidden_size, bottleneck_dim, bias=False))
        self.norm = RMSNorm(bottleneck_dim)
        self.sep_token_id = sep_token_id

    def forward(self, token_ids: Tensor, attention_mask: Tensor, token_clusters: Tensor) -> Tensor:
        token_embeddings = self.core(token_ids, attention_mask=attention_mask)['last_hidden_state']
        sparsity_mask = (attention_mask == 1).bitwise_and(token_ids != self.sep_token_id)
        out = self.aggregator.forward(token_embeddings[sparsity_mask], token_clusters[sparsity_mask])
        return self.norm(out)
