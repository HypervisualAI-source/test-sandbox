import torch
import torch.nn as nn
from torch.nn import Module, ModuleList
from einops import rearrange, repeat                                                                                                                          
from einops.layers.torch import Rearrange
                                                                                                    
                                              
class HVAI_MultiheadLatentAttention(Module):
    """
    Args:
        d_latent: the dimension of the latent subspace used to compute attention
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn`` and ``to_out``. Default: ``0.0`` (no dropout).
    """
    
    def __init__(self, d_latent, embed_dim, num_heads, dropout = 0.):
        super().__init__()                                                
        
        self.d_latent = d_latent
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = d_latent ** -0.5
        self.head_dim = embed_dim // num_heads

        self.norm = nn.LayerNorm(embed_dim)                                                                
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(embed_dim, num_heads * self.head_dim, bias = True)
        self.kv_down_proj = nn.Linear(embed_dim, d_latent, bias = True)
        
        self.w_uk = nn.Linear(d_latent, num_heads * self.head_dim, bias = False)
        self.w_uv = nn.Linear(d_latent, num_heads * self.head_dim, bias = False)
                                                                                                                
        self.to_out = nn.Sequential(                                                                                                                         
            nn.Linear(num_heads * self.head_dim, embed_dim),
            nn.Dropout(dropout)
        )                         
                                                                                            
    def forward(self, x):                          
        
        """                                                       
        Args:                                              
            x: [batch_szie, token_size, embed_dim]                                                                                          
                - batch_szie: How many independent samples (sequences) the model processes at the same time in one forward pass.
                - token_size: How many tokens are in one input sequence.
                - embed_dim: The size of the vector representing each token.
        """
        b, n, _ = x.shape
        
        # --- input normalization ---
        x = self.norm(x)     

        # --- KV compression ---
        latent_kv = self.kv_down_proj(x)

        # --- query preparation ---
        q = self.to_q(x)                                               
        # 'b n (h d) -> b h n d': b = batch size, n = token size, h = the number of head, d = the dimension of head 
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.num_heads)                                                
                                                                                                                                                                            
        # --- matrix absorption ---                                                                                                          
        w_uk_res = self.w_uk.weight.view(self.num_heads, self.head_dim, self.d_latent)         
        # 'bhnd,hdl->bhnl': b = batch size, h = number of head, n = token size, d = dimension of head, l = the dimension of the latent subspace          
        q_latent = torch.einsum('bhnd,hdl->bhnl', q, w_uk_res)                                                                 

        # --- attention score (in latent space) ---                                                                                                                                       
        dots = torch.matmul(q_latent, latent_kv.unsqueeze(1).transpose(-1, -2)) * self.scale
        # softmax function
        attn = self.attend(dots)                                                
        attn = self.dropout(attn)                                                                                 
                                                                                                                                   
        # --- value aggregation & expansion ---                           
        # 1. weighted sum in latent space: [b, h, n, d_latent]                                                                          
        weighted_latent = torch.matmul(attn, latent_kv.unsqueeze(1))
        # 2. Expand back to head dimension: [b, n, (h d)]         
        w_uv_res = self.w_uv.weight.view(self.num_heads, self.head_dim, self.d_latent)    
        # 'bhnl,hdl->bnhd': b = batch size, h = the number of head, n = token size, l = the dimension of the latent subspace, d = the dimension of head        
        out = torch.einsum('bhnl,hdl->bnhd', weighted_latent, w_uv_res) 
        # 'b n h d -> b n (h d)': b = batch size, n = token size, h = the number of head, d = the dimension of head
        out = rearrange(out, 'b n h d -> b n (h d)')                                                                                                                               

        return self.to_out(out)