import torch
import torch.nn as nn
import loralib as lora


def inject_lora_into_clip_vit(model, rank=4, dropout=0.05, target_modules=['attn']):
    """
    Inject LoRA into CLIP ViT Transformer blocks.
    
    Args:
        model: CLIP_EBC model
        rank: LoRA rank (4, 8, or 16)
        target_modules: Which modules to apply LoRA to
    
    Returns:
        model with LoRA injected
    """
    layers_replaced = 0
    
    # Access the transformer blocks
    if hasattr(model, 'image_encoder'):
        if hasattr(model.image_encoder, 'transformer'):
            # This is a ViT model
            for block_idx, block in enumerate(model.image_encoder.transformer.resblocks):
                if hasattr(block, 'attn') and isinstance(block.attn, nn.MultiheadAttention):
                    print(f"Replacing attention in block {block_idx}")
                    
                    d_model = block.attn.embed_dim
                    n_head = block.attn.num_heads
                    
                    # Create LoRA-enhanced attention
                    new_attn = LoRAMultiheadAttention(d_model, n_head, rank=rank, dropout=dropout)
                    
                    # Copy pre-trained weights
                    new_attn.in_proj.weight.data = block.attn.in_proj_weight.data.clone()
                    if block.attn.in_proj_bias is not None:
                        new_attn.in_proj.bias.data = block.attn.in_proj_bias.data.clone()
                    new_attn.out_proj.weight.data = block.attn.out_proj.weight.data.clone()
                    if block.attn.out_proj.bias is not None:
                        new_attn.out_proj.bias.data = block.attn.out_proj.bias.data.clone()
                    
                    # Replace
                    block.attn = new_attn
                    layers_replaced += 1
    
    print(f"Replaced {layers_replaced} attention layers with LoRA (rank={rank})")
    return model


class LoRAMultiheadAttention(nn.Module):
    """
    LoRA-enhanced MultiheadAttention for CLIP ViT.
    Compatible with the forward signature used in ResidualAttentionBlock.
    """
    def __init__(self, embed_dim, num_heads, rank=4, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Use LoRA for Q, V projections only (standard practice)
        self.in_proj = lora.MergedLinear(
            embed_dim,
            3 * embed_dim,
            r=rank,
            enable_lora=[True, False, True],  # Q, K, V
            bias=True
        )
        
        self.out_proj = lora.Linear(embed_dim, embed_dim, r=rank, bias=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, x_for_key=None, x_for_value=None, 
                need_weights=False, attn_mask=None):
        """
        Forward pass compatible with CLIP's ResidualAttentionBlock.
        Args:
            x: query tensor [L, N, D] where L=seq_len, N=batch, D=embed_dim
        """
        if x_for_key is None:
            x_for_key = x
        if x_for_value is None:
            x_for_value = x
        
        tgt_len, bsz, embed_dim = x.size()
        
        # Project Q, K, V
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = torch.bmm(q, k.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.bmm(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            return attn_output, attn_weights
        return attn_output, None


def prepare_model_for_lora_training(model, device='cuda'):
    """
    Freeze all parameters except LoRA, VPT, and task-specific heads.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters and move to device
    lora.mark_only_lora_as_trainable(model)
    
    # Move LoRA parameters to device
    for name, module in model.named_modules():
        if isinstance(module, (lora.Linear, lora.MergedLinear)):
            module.to(device)
    
    # Unfreeze VPT tokens
    vpt_params = 0
    for name, param in model.named_parameters():
        if 'vpt' in name.lower():
            param.requires_grad = True
            vpt_params += param.numel()

    if vpt_params > 0:
        print(f"   VPT parameters: {vpt_params:,}")
    
    # Decoder stays FROZEN to prevent catastrophic forgetting
    # (10.6M params — unfreezing it destroyed QNRF knowledge)

    # Unfreeze task-specific layers
    if hasattr(model, 'projection'):
        for param in model.projection.parameters():
            param.requires_grad = True
    
    # Logit scale
    if hasattr(model, 'logit_scale'):
        model.logit_scale.requires_grad = True
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    
    return model
