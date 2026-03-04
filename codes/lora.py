import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

from .config import LoRAFineTuningConfig


def apply_decoder_lora(decoder: nn.Module, config: LoRAFineTuningConfig) -> nn.Module:
    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=list(config.target_modules),
    )
    peft_decoder = get_peft_model(decoder, lora_config)

    for parameter in peft_decoder.parameters():
        parameter.requires_grad = False

    for name, parameter in peft_decoder.named_parameters():
        if "lora_" in name or name.startswith("lm_head"):
            parameter.requires_grad = True

    return peft_decoder
