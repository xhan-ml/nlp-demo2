import torch
import torch.nn as nn
from transformers import BertModel

class BertNERModel(nn.Module):
    def __init__(self, pretrain_name, num_labels, dropout: float = 0.1):
        super().__init__()
       
        self.bert = BertModel.from_pretrained(pretrain_name)
        self.dropout = nn.Dropout(dropout)
       
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
       
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        seq_out = bert_out.last_hidden_state  # shape: [batch, seq_len, 768]
        seq_out = self.dropout(seq_out)
        logits = self.classifier(seq_out)     # shape: [batch, seq_len, num_labels]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
           
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        return loss, logits
