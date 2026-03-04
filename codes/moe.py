import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyTopKRouter(nn.Module):
    def __init__(self, embedding_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.route = nn.Linear(embedding_dim, num_experts)
        self.noise = nn.Linear(embedding_dim, num_experts)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.route(hidden_states)
        noise_scale = F.softplus(self.noise(hidden_states))
        noisy_logits = logits + torch.randn_like(logits) * noise_scale
        top_k_logits, top_k_indices = noisy_logits.topk(self.top_k, dim=-1)
        sparse_logits = torch.full_like(noisy_logits, float("-inf")).scatter(-1, top_k_indices, top_k_logits)
        return F.softmax(sparse_logits, dim=-1), top_k_indices


class ExpertFFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_states)


class MoELayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        top_k: int,
        expert_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim
        hidden_dim = expert_hidden_dim or input_dim * 4
        self.router = NoisyTopKRouter(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList(
            [ExpertFFN(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flattened = hidden_states.reshape(-1, hidden_dim)
        gate_weights, top_k_indices = self.router(flattened)

        combined = torch.zeros(
            flattened.size(0),
            self.output_dim,
            device=flattened.device,
            dtype=flattened.dtype,
        )
        expanded_inputs = flattened.repeat_interleave(self.top_k, dim=0)
        flat_expert_indices = top_k_indices.reshape(-1)

        for expert_id, expert in enumerate(self.experts):
            positions = torch.where(flat_expert_indices == expert_id)[0]
            if positions.numel() == 0:
                continue
            expert_output = expert(expanded_inputs[positions])
            token_indices = positions // self.top_k
            expert_weights = gate_weights[token_indices, flat_expert_indices[positions]].unsqueeze(-1)
            combined.index_add_(0, token_indices, expert_output * expert_weights)

        return combined.view(batch_size, sequence_length, self.output_dim), gate_weights


def load_balancing_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    num_tokens, num_experts = gate_weights.shape
    token_fraction = gate_weights.sum(dim=0) / num_tokens
    mean_probability = gate_weights.mean(dim=0)
    return num_experts * torch.sum(token_fraction * mean_probability)


class TransformerBlockWithMoE(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.moe = MoELayer(embedding_dim, embedding_dim, num_experts, top_k)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        attended, _ = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
        )
        hidden_states = self.norm1(hidden_states + self.dropout(attended))
        moe_output, gate_weights = self.moe(hidden_states)
        hidden_states = self.norm2(hidden_states + self.dropout(moe_output))
        return hidden_states, gate_weights
