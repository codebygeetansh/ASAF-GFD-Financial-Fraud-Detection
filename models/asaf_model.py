import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveSelfAttentionFusion(nn.Module):
    """
    Synopsis Section 3.1 & 3.3 (Novel Contribution):
    Computes node-specific attention scores:
    e_1i = tanh(W_a * Z_1i + b)
    e_2i = tanh(W_a * Z_2i + b)
    alpha_1i, alpha_2i = Softmax(e_1i, e_2i)
    Z_i = alpha_1i * Z_1i + alpha_2i * Z_2i
    """
    def __init__(self, embed_dim):
        super(AdaptiveSelfAttentionFusion, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Shared learnable projection W_a and bias b (Synopsis Eq.)
        self.W_a = nn.Linear(embed_dim, embed_dim)
        self.v_a = nn.Linear(embed_dim, 1, bias=False)
        
        # Context calibration gate for dual-path synergy
        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Sigmoid()
        )

    def forward(self, Z1, Z2, structural_energy=None):
        Z1_n = self.norm1(Z1)
        Z2_n = self.norm2(Z2)

        # Shared attention projection: e_1i = tanh(W_a Z_1i + b), e_2i = tanh(W_a Z_2i + b)
        e1 = self.v_a(torch.tanh(self.W_a(Z1_n)))
        e2 = self.v_a(torch.tanh(self.W_a(Z2_n)))

        # Incorporate local high-frequency structural anomaly energy if available
        if structural_energy is not None:
            e1 = e1 + 0.8 * structural_energy
            e2 = e2 - 0.8 * structural_energy

        scores = torch.cat([e1, e2], dim=-1)
        weights = F.softmax(scores, dim=-1)

        alpha_1 = weights[:, 0].unsqueeze(-1)
        alpha_2 = weights[:, 1].unsqueeze(-1)

        # Weighted node-level fusion: Z_i = alpha_1 * Z_1i + alpha_2 * Z_2i
        Z_attentive = (alpha_1 * Z1_n) + (alpha_2 * Z2_n)
        
        gate = self.context_gate(torch.cat([Z1_n, Z2_n], dim=-1))
        Z_fused = Z_attentive + gate * Z1_n + (1.0 - gate) * Z2_n

        return Z_fused, alpha_1, alpha_2


class DualPathFraudGNN(nn.Module):
    def __init__(self, in_features, num_bands=3, embed_dim=64, hidden_dim=32, fusion_type='attention'):
        super(DualPathFraudGNN, self).__init__()
        self.fusion_type = fusion_type

        self.path1_encoder = nn.Sequential(
            nn.Linear(in_features * num_bands, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.PReLU(),
            nn.Dropout(0.15)
        )

        self.path2_encoder = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.PReLU(),
            nn.Dropout(0.15)
        )

        if fusion_type == 'attention':
            self.attention_fusion = AdaptiveSelfAttentionFusion(embed_dim)
            mlp_in_dim = embed_dim
        else:
            mlp_in_dim = embed_dim * 2

        # MLP for Nonlinear Mapping: H = MLP(Z) (Base Paper Eq. 13)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU()
        )

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, Z1_concat, Z2_lowpass, structural_energy=None):
        Z1 = self.path1_encoder(Z1_concat)
        Z2 = self.path2_encoder(Z2_lowpass)

        if self.fusion_type == 'attention':
            Z_fused, alpha_1, alpha_2 = self.attention_fusion(Z1, Z2, structural_energy)
        else:
            Z_fused = torch.cat([Z1, Z2], dim=-1)
            alpha_1 = torch.full((Z1.shape[0], 1), 0.5, device=Z1.device)
            alpha_2 = torch.full((Z2.shape[0], 1), 0.5, device=Z2.device)

        H = self.mlp(Z_fused)
        logits = self.classifier(H)

        return logits, H, Z1, Z2, alpha_1, alpha_2