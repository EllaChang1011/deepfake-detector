from peft import LoraConfig

def get_clip_lora_config(r=8, alpha=16, dropout=0.05):
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["self_attn.k_proj", "self_attn.v_proj"],
        task_type="FEATURE_EXTRACTION"
    )
