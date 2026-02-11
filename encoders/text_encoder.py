import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    """
    Text encoder using BERT. Converts text commands to embeddings for fusion.
    """
    def __init__(self, model_name: str = 'bert-base-uncased', output_dim: int = 512, device: str = 'cpu'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.device = device
        self.output_dim = output_dim
        self.proj = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.to(device)

    def forward(self, texts):
        # texts: list of strings
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.bert(**encoded)
        pooled = outputs.pooler_output  # (batch, hidden_size)
        return self.proj(pooled)  # (batch, output_dim)
