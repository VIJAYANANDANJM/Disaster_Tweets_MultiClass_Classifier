import os

import torch.nn as nn
from transformers import AutoConfig, AutoModel, BertConfig

class DeLTran15(nn.Module):
    def __init__(self, model_name, num_classes=5, local_files_only=False):
        super().__init__()
        self.encoder = self._load_encoder(
            model_name=model_name,
            local_files_only=local_files_only
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            num_classes
        )

    @staticmethod
    def _offline_minilm_config():
        """Config that matches sentence-transformers/all-MiniLM-L6-v2."""
        return BertConfig(
            vocab_size=30522,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=1536,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            model_type="bert"
        )

    @classmethod
    def _load_encoder(cls, model_name, local_files_only=False):
        """
        Load the encoder from pretrained weights when available.
        During offline inference, fall back to a local MiniLM config and rely on
        the saved project checkpoint to populate the weights afterward.
        """
        try:
            return AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only
            )
        except Exception:
            config = None

            if os.path.isdir(model_name):
                try:
                    config = AutoConfig.from_pretrained(
                        model_name,
                        local_files_only=True
                    )
                except Exception:
                    config = None

            if config is None:
                config = cls._offline_minilm_config()

            return AutoModel.from_config(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_embedding)
        return self.classifier(x)
