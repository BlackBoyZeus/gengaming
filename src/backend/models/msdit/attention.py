# External imports with versions
import torch  # torch ^2.0.0
import torch.nn as nn  # torch ^2.0.0
from einops import rearrange, repeat  # einops ^0.6.0
import math  # math ^3.9.0

# Internal imports
from models.msdit.config import MSDiTConfig

# Global constants
MAX_SEQUENCE_LENGTH = 102
ATTENTION_HEAD_DIM = 64
MEMORY_EFFICIENT_ATTENTION = True
FREEBSD_GPU_FALLBACK = True

@torch.jit.script
def apply_rotary_embeddings(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embeddings to input tensor."""
    x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
    return x * cos + x2 * sin

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention implementation with FreeBSD-optimized tensor operations 
    and memory-efficient attention patterns.
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int, 
        dropout: float = 0.1,
        use_memory_efficient_attention: bool = MEMORY_EFFICIENT_ATTENTION
    ):
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"Hidden dimension {hidden_dim} must be divisible by number of heads {num_heads}")
            
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_memory_efficient_attention = use_memory_efficient_attention
        
        # Initialize projection layers with FreeBSD optimizations
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize rotary embeddings cache
        self.register_buffer(
            "rotary_emb_cache",
            self._create_rotary_embeddings(MAX_SEQUENCE_LENGTH),
            persistent=False
        )

    def _create_rotary_embeddings(self, seq_length: int) -> torch.Tensor:
        """Creates cached rotary embeddings for position encoding."""
        position = torch.arange(seq_length).float()
        dim = self.head_dim // 2
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        
        return torch.stack([
            torch.sin(sinusoid_inp),
            torch.cos(sinusoid_inp)
        ], dim=0)

    @torch.cuda.amp.autocast(enabled=True)
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass with memory-efficient attention computation.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, hidden_dim]
            key: Key tensor of shape [batch_size, seq_len_k, hidden_dim]
            value: Value tensor of shape [batch_size, seq_len_v, hidden_dim]
            mask: Optional attention mask tensor
            
        Returns:
            Output tensor of shape [batch_size, seq_len_q, hidden_dim]
        """
        batch_size, seq_len_q = query.shape[:2]
        seq_len_k = key.shape[1]
        
        # Project inputs
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        
        # Reshape for multi-head attention
        query = rearrange(query, 'b n (h d) -> b h n d', h=self.num_heads)
        key = rearrange(key, 'b n (h d) -> b h n d', h=self.num_heads)
        value = rearrange(value, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Apply rotary embeddings
        sin, cos = self.rotary_emb_cache[:, :seq_len_q], self.rotary_emb_cache[:, :seq_len_k]
        query = apply_rotary_embeddings(query, cos, sin)
        key = apply_rotary_embeddings(key, cos, sin)
        
        if self.use_memory_efficient_attention and FREEBSD_GPU_FALLBACK:
            # Memory-efficient attention implementation
            scale = 1 / math.sqrt(self.head_dim)
            
            # Compute attention scores in chunks
            chunk_size = min(seq_len_k, 1024)
            attention_output = torch.zeros_like(query)
            
            for chunk_start in range(0, seq_len_k, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_len_k)
                
                # Compute attention scores for current chunk
                scores = torch.matmul(query, key[..., chunk_start:chunk_end, :].transpose(-2, -1)) * scale
                
                if mask is not None:
                    chunk_mask = mask[:, :, :, chunk_start:chunk_end]
                    scores = scores.masked_fill(~chunk_mask, float('-inf'))
                
                attn_weights = torch.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                # Update attention output
                attention_output += torch.matmul(attn_weights, value[..., chunk_start:chunk_end, :])
        else:
            # Standard attention implementation
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))
            
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attention_output = torch.matmul(attn_weights, value)
        
        # Combine heads and project output
        output = rearrange(attention_output, 'b h n d -> b n (h d)')
        return self.output_proj(output)

    def extra_repr(self) -> str:
        """Returns extra representation string for debugging."""
        return f'hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}'