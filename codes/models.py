import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

from .config import ClassifierConfig, GeneratorConfig
from .moe import TransformerBlockWithMoE


class SparseMoEClassifier(nn.Module):
    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__()
        base_model = AutoModel.from_pretrained(config.encoder_name)
        self.embedding = base_model.get_input_embeddings()
        embedding_dim = self.embedding.embedding_dim
        self.block = TransformerBlockWithMoE(
            embedding_dim=embedding_dim,
            num_heads=config.num_heads,
            num_experts=config.num_experts,
            top_k=config.top_k,
            dropout=config.dropout,
        )
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embedding_dim, config.num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embedding(input_ids)
        hidden_states, gate_weights = self.block(hidden_states, attention_mask=attention_mask)
        pooled = self.pooler(hidden_states.transpose(1, 2)).squeeze(-1)
        return self.classifier(pooled), gate_weights


class MoEEncoder(nn.Module):
    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        base_model = AutoModel.from_pretrained(config.encoder_name)
        self.embedding = base_model.get_input_embeddings()
        embedding_dim = self.embedding.embedding_dim
        self.block = TransformerBlockWithMoE(
            embedding_dim=embedding_dim,
            num_heads=config.num_heads,
            num_experts=config.num_experts,
            top_k=config.top_k,
            dropout=config.dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embedding(input_ids)
        return self.block(hidden_states, attention_mask=attention_mask)


class ModernBertMoET5Generator(nn.Module):
    def __init__(self, config: GeneratorConfig, encoder: MoEEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder or MoEEncoder(config)
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
        self.decoder = T5ForConditionalGeneration.from_pretrained(config.decoder_name)
        self.decoder_tokenizer = T5Tokenizer.from_pretrained(config.decoder_name, legacy=False)

        encoder_dim = self.encoder.embedding.embedding_dim
        decoder_dim = self.decoder.config.d_model
        if encoder_dim == decoder_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Linear(encoder_dim, decoder_dim),
                nn.Tanh(),
                nn.LayerNorm(decoder_dim),
            )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        encoder_output, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_output = self.projection(encoder_output)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_output)
        return self.decoder(
            encoder_outputs=encoder_outputs,
            labels=labels,
            use_cache=False,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        num_beams: int = 4,
        **kwargs,
    ) -> torch.Tensor:
        encoder_output, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_output = self.projection(encoder_output)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_output)
        return self.decoder.generate(
            encoder_outputs=encoder_outputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **kwargs,
        )

    def generate_text(self, text: str, max_new_tokens: int = 64, **kwargs) -> str:
        encoded = self.encoder_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        encoded = {key: value.to(next(self.parameters()).device) for key, value in encoded.items()}
        generated = self.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        return self.decoder_tokenizer.decode(generated[0], skip_special_tokens=True)
