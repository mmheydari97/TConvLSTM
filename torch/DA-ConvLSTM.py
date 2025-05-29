import torch
import torch.nn as nn

class DurationAwareConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, duration_feature_dim=16):
        super(DurationAwareConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Convolution for input and hidden states (concatenated) to gates
        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=4 * hidden_dim, # i, f, o, c_tilde
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        # Small MLP to process duration
        self.duration_mlp = nn.Sequential(
            nn.Linear(1, duration_feature_dim),
            nn.ReLU(),
            nn.Linear(duration_feature_dim, hidden_dim) # Output matches hidden_dim for scaling, or 1 for a single factor
        )
        # Or a simpler direct scaling factor learnable parameter
        # self.duration_scale_param = nn.Parameter(torch.ones(1))


    def forward(self, input_tensor, current_duration, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # Concatenate along channel axis
        gates = self.conv_gates(combined) # B, 4*hidden_dim, H, W

        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g) # This is c_tilde

        # Process duration
        # Ensure duration is [B, 1]
        current_duration_reshaped = current_duration.unsqueeze(-1).float()
        # Option 1: MLP generates a scaling factor per channel (or a single factor to broadcast)
        # This example assumes output matches hidden_dim and is used as a channel-wise scale
        # duration_factor_channels = 1.0 + torch.relu(self.duration_mlp(current_duration_reshaped)).unsqueeze(-1).unsqueeze(-1) # B, hidden_dim, 1, 1
        # g_modified = g * duration_factor_channels

        # Option 2: Simpler - MLP generates a single scalar factor, then ensure it's positive
        # Example: make MLP output a single value, then apply softplus or exp
        duration_scalar_effect = self.duration_mlp(current_duration_reshaped) # B, hidden_dim
        # To make it a single scalar factor that broadcasts:
        # Assume duration_mlp outputs a single value: nn.Linear(duration_feature_dim, 1)
        # duration_factor = torch.exp(duration_scalar_effect * 0.1).unsqueeze(-1).unsqueeze(-1) # B, 1, 1, 1 -- apply scaling factor to avoid large values
        # Or more simply, if MLP outputs one value:
        duration_factor = 1.0 + torch.relu(self.duration_mlp(current_duration_reshaped)).mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1) # B,1,H,W
        # A simple log-based factor:
        # duration_factor = (torch.log(current_duration.float() + 1e-6) + 1.0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


        g_modified = g * duration_factor # Element-wise multiplication

        c_next = f * c_cur + i * g_modified
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_gates.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_gates.weight.device))

# To use it for a sequence:
# rnn_layer = DurationAwareConvLSTMCell(...)
# inputs = [(frame1, duration1), (frame2, duration2), ...] # Your preprocessed data
# h, c = rnn_layer.init_hidden(batch_size, image_size)
# outputs = []
# for frame_input, duration_input in zip(frames_sequence, durations_sequence): # frame_input: B, C, H, W; duration_input: B
#     h, c = rnn_layer(frame_input, duration_input, (h,c))
#     outputs.append(h)
