import torch                                       
import torch.nn as nn
# importing multi-head latent attention: HVAI_MultiheadLatentAttention
from hvai_nn import HVAI_MultiheadLatentAttention
                                              
"""                                                                                                              
Args:                           
    batch_szie: How many independent samples (sequences) the model processes at the same time in one forward pass.
    token_size: How many tokens are in one input sequence.
    embed_dim: The size of the vector representing each token.
    d_latent: The dimension of the latent subspace used to compute attention
    num_heads: The number of parallel attention heads.
"""                                                                                                                                                                                                             
                                                                                                                                                                  
# input data 
input_test = torch.randn(1, 1000, 1024)   # (1, 1000, 1024): [token_size, batch_size, embed_dim]
                                                                                                                                                            
# case 1: multi-head attention                             
mha_attention = nn.MultiheadAttention(embed_dim = 1024, num_heads = 16, batch_first=True)  
with torch.no_grad():
    out_test, _ = mha_attention(input_test, input_test, input_test)  # (input_test, input_test, input_test): (q, k, v) (q = k = v)
    print("out_test:", out_test.shape)
        
# case 2: multi-head latent attention                    
# monkey patch
nn.MultiheadAttention = HVAI_MultiheadLatentAttention                                             
hvai_mla = nn.MultiheadAttention(d_latent = 20, embed_dim = 1024, num_heads = 16)             

with torch.no_grad():
    out_test = hvai_mla(input_test)
    print("out_test:", out_test.shape)
