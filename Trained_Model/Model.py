import torch.nn as nn
from transformers import AutoModel

class DeLTran15(nn.Module):
    def __init__(self, model_name, num_classes=5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            num_classes
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_embedding)
        return self.classifier(x)
