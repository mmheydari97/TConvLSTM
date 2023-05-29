import torch
import torch.nn as nn


class TimeAwareConvLSTMCell(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size):
        super(TimeAwareConvLSTMCell, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding)           

        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.decay_factor = nn.Parameter(torch.Tensor(1))

    def forward(self, X, H_prev, C_prev, duration):
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        decay_factor = torch.exp(-self.decay_factor * duration)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev ) * decay_factor
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)
        output_gate = torch.sigmoid(o_conv + self.W_co * C )
        H = output_gate * self.activation(C)

        return H, C
    


class TConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size, return_sequence=False):
        super(TConvLSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.out_channels = out_channels
        self.return_sequence = return_sequence
        
        self.cell = TimeAwareConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size)

    def forward(self, X, durations):
        batch_size, seq_len, channels, height, width = X.size()
        
        output = torch.zeros(batch_size, seq_len, self.out_channels,  
        height, width, device=self.device)
        
        H = torch.zeros(batch_size, self.out_channels, 
        height, width, device=self.device)
        
        C = torch.zeros(batch_size,self.out_channels, 
        height, width, device=self.device)

        for t, duration in enumerate(durations):
            H, C = self.cell(X[:, t], H, C, duration)
            output[:, t, ...] = H
        
        if not self.return_sequence:
            output = torch.squeeze(output[:, -1, ...], dim=1)
        
        return output

class TimeAwareConvLSTMAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, padding, activation, frame_size):
        super(TimeAwareConvLSTMAutoencoder, self).__init__()
        self.encoder = TimeAwareConvLSTMEncoder(in_channels, hidden_channels, kernel_size, padding, activation, frame_size)
        self.decoder_lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.decoder_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size, padding=padding)

    def forward(self, X, durations):
        H, C = self.encoder(X, durations)
        H = H.unsqueeze(1)
        H, _ = self.decoder_lstm(H)
        H = H.squeeze(1)
        reconstructed_X = self.decoder_conv(H)
        return reconstructed_X