import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualAttention(nn.Module):
    """Attention with a residual skip connection to preserve BiLSTM signal magnitude.
    
    Standard multiplicative attention crushes signal by 1/seq_len (~4%).
    The residual form preserves 100% of the original signal while adding
    a bounded [0, 2x] attention-modulated refinement on top.
    """
    def __init__(self, hidden_size):
        super(ResidualAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)  # [Batch, Seq, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)  # [Batch, Seq] — bounded, sums to 1.0
        # Residual: full signal preserved + softmax-bounded refinement
        attended = encoder_outputs + encoder_outputs * weights.unsqueeze(-1)  # [Batch, Seq, Hidden]
        return attended

class SpatioTemporalHybridNet(nn.Module):
    def __init__(self, n_dynamic=10, n_static=6, seq_length=24, pred_length=24):
        super(SpatioTemporalHybridNet, self).__init__()
        
        self.n_dynamic = n_dynamic
        self.n_static = n_static
        
        # 1D-CNN Block (Processes ONLY dynamic temporal features)
        self.conv1d = nn.Conv1d(in_channels=n_dynamic, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(0.2)
        
        # BiLSTM Block (with inter-layer dropout for regularization)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, 
                              batch_first=True, bidirectional=True, dropout=0.2)
        
        # Residual Self-Attention (preserves BiLSTM signal)
        self.attention = ResidualAttention(hidden_size=256)
        
        # Late-Fusion Nonlinear Regressor Head (Time-Distributed)
        # Input per timestep: 256 (temporal) + n_static (static features)
        fused_dim = 256 + n_static
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # ===== DUAL-PATH FEATURE ROUTING =====
        # Dynamic temporal features → CNN/BiLSTM path
        dynamic = x[:, :, :self.n_dynamic]       # [Batch, 24, 9]
        # Static contextual features → Late fusion (constant across 24h, take one row)
        static = x[:, 0, self.n_dynamic:self.n_dynamic + self.n_static]  # [Batch, 6]
        # Daily baseline → Skip connection
        daily_baseline = x[:, 0, -1]              # [Batch]
        
        # ===== TEMPORAL PIPELINE =====
        # CNN expects [Batch, Channels, Length]
        x_in = dynamic.permute(0, 2, 1)          # [Batch, 9, 24]
        
        x_in = self.conv1d(x_in)
        x_in = self.batch_norm(x_in)
        x_in = self.relu(x_in)
        x_in = self.dropout_cnn(x_in)
        
        x_in = x_in.permute(0, 2, 1)             # [Batch, 24, 64]
        
        lstm_out, _ = self.bilstm(x_in)           # [Batch, 24, 256]
        attended_out = self.attention(lstm_out)    # [Batch, 24, 256] — signal preserved!
        
        # ===== LATE FUSION (Time-Distributed) =====
        # Expand static features to match the 24 timesteps
        static_expanded = static.unsqueeze(1).expand(-1, 24, -1)  # [Batch, 24, 6]
        
        # Concatenate temporal context with static features per timestep
        combined = torch.cat([attended_out, static_expanded], dim=-1)  # [Batch, 24, 262]
        
        # Predict ratio per timestep
        ratio = F.softplus(self.regressor(combined)).squeeze(-1)  # [Batch, 24]
        
        # ===== MULTIPLICATIVE DAILY CONSTRAINT =====
        # Multiply by daily baseline (both in raw physical space)
        final_output = ratio * daily_baseline.unsqueeze(-1)  # [Batch, 24]
        
        return final_output

