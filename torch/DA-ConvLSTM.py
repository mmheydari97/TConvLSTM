import torch
import torch.nn as nn
import torch.nn.functional as F

class DurationAwareConvLSTMCellPyTorch(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True,
                 duration_mode="log", batch_norm_durations=False,
                 duration_feature_dim=16, exp_damping_factor=0.1, epsilon=1e-6):
        super(DurationAwareConvLSTMCellPyTorch, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        self.bias = bias
        self.epsilon = epsilon

        # Duration processing settings
        if duration_mode not in ["log", "exp"]:
            raise ValueError("duration_mode must be 'log' or 'exp'")
        self.duration_mode = duration_mode
        self.batch_norm_durations = batch_norm_durations
        self.exp_damping_factor = exp_damping_factor

        # Convolution for input and hidden states (concatenated) to gates
        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=4 * hidden_dim, # i, f, o, g
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        # Small MLP to process duration
        self.duration_mlp = nn.Sequential(
            nn.Linear(1, duration_feature_dim),
            nn.ReLU(),
            nn.Linear(duration_feature_dim, 1) # Output a single scalar effect
        )

    def forward(self, input_tensor, current_duration, cur_state):
        # input_tensor: (B, C_in, H, W)
        # current_duration: (B,)
        # cur_state: (h_cur, c_cur) each (B, C_hidden, H, W)
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.conv_gates(combined) # B, 4*hidden_dim, H, W

        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g) # Candidate cell state (c_tilde)

        # --- Duration Processing ---
        d_val = current_duration.float() # Shape: [B]

        if self.batch_norm_durations:
            # Normalize across the batch for the current timestep
            mean_d = d_val.mean(dim=0, keepdim=True)
            std_d = d_val.std(dim=0, keepdim=True)
            d_val = (d_val - mean_d) / (std_d + self.epsilon)

        mlp_input = d_val.unsqueeze(-1) # Shape: [B, 1]
        scalar_effect = self.duration_mlp(mlp_input) # Shape: [B, 1]

        if self.duration_mode == "log":
            # Additive impact, factor >= 1.0
            duration_factor = 1.0 + F.relu(scalar_effect)
        elif self.duration_mode == "exp":
            # Exponential impact
            duration_factor = torch.exp(scalar_effect * self.exp_damping_factor)
        
        # Reshape factor for broadcasting: [B, 1, 1, 1]
        factor_reshaped = duration_factor.view(duration_factor.shape[0], 1, 1, 1)
        
        g_modified = g * factor_reshaped
        # --- End Duration Processing ---

        c_next = f * c_cur + i * g_modified
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = self.conv_gates.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

# Example of a layer using the cell (manual unrolling)
class DurationConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size,
                 duration_mode="log", batch_norm_durations=False,
                 duration_feature_dim=16, exp_damping_factor=0.1):
        super(DurationConvLSTM, self).__init__()
        self.cell = DurationAwareConvLSTMCellPyTorch(
            input_dim, hidden_dim, kernel_size,
            duration_mode=duration_mode,
            batch_norm_durations=batch_norm_durations,
            duration_feature_dim=duration_feature_dim,
            exp_damping_factor=exp_damping_factor
        )
        self.hidden_dim = hidden_dim

    def forward(self, frames_sequence, durations_sequence, initial_hidden_state=None):
        # frames_sequence: (B, SeqLen, C_in, H, W)
        # durations_sequence: (B, SeqLen)
        b, seq_len, _, h, w = frames_sequence.shape
        
        if initial_hidden_state is None:
            h_state, c_state = self.cell.init_hidden(b, (h, w))
        else:
            h_state, c_state = initial_hidden_state
            
        outputs = []
        for t in range(seq_len):
            frame_t = frames_sequence[:, t, :, :, :]    # (B, C_in, H, W)
            duration_t = durations_sequence[:, t]       # (B,)
            h_state, c_state = self.cell(frame_t, duration_t, (h_state, c_state))
            outputs.append(h_state)
            
        return torch.stack(outputs, dim=1), (h_state, c_state) # (B, SeqLen, C_hidden, H, W)
    
# To use it for a sequence:
# rnn_layer = DurationAwareConvLSTMCell(...)
# inputs = [(frame1, duration1), (frame2, duration2), ...] # Your preprocessed data
# h, c = rnn_layer.init_hidden(batch_size, image_size)
# outputs = []
# for frame_input, duration_input in zip(frames_sequence, durations_sequence): # frame_input: B, C, H, W; duration_input: B
#     h, c = rnn_layer(frame_input, duration_input, (h,c))
#     outputs.append(h)
