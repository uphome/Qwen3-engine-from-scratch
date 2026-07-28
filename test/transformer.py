import torch
import math
import torch.nn as nn

class self_attention(nn.Module):
    def __init__(self,embedding):
        super().__init__()  
        self.embedding =embedding
        self.Linear_q=nn.Linear(embedding,embedding)
        self.Linear_k=nn.Linear(embedding,embedding)
        self.Linear_v=nn.Linear(embedding,embedding)
    
    def forward(self,x):
        q = self.Linear_q(x)
        v = self.Linear_v(x)
        k = self.Linear_k(x)

        k_t =k.transpose(-2,-1)

        s=torch.matmul(q,k_t)

        s = s/math.sqrt(self.embedding)

        att_weight= torch.softmax(s,dim=-1)

        o = torch.matmul(att_weight,v)

        return o 
    
batch_size = 2
seq_len = 5
embed_dim = 16  # 嵌入维度，与构造函数中的 embedding 一致

# 创建模块实例
attn = self_attention(embedding=embed_dim)

# 生成随机输入: (batch_size, seq_len, embed_dim)
x = torch.randn(batch_size, seq_len, embed_dim)

# 前向传播
output = attn(x)

print("输入形状:", x.shape)        # torch.Size([2, 5, 16])
print("输出形状:", output.shape)  # torch.Size([2, 5, 16])