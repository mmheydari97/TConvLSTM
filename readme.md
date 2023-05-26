# TimeAwareConvLSTMCell

This is an implementation of TimeAwareConvLSTMCell, a variant of the Convolutional LSTM cell that incorporates a time-aware decay factor. The TimeAwareConvLSTMCell is suitable for modeling spatiotemporal data more efficiently and is commonly used in tasks such as video processing, action recognition, and motion prediction.

## Implementation Details

The `TimeAwareConvLSTMCell` class is implemented using `tf.keras.layers.Layer` in TensorFlow or `nn.Module` in PyTorch. It extends the those classes and provides a custom implementation of the Convolutional LSTM cell with time-aware decay.

The key components of the `TimeAwareConvLSTMCell` include:
- Convolutional layer: Performs a 2D convolution on the input and hidden state.
- Weight matrices: `W_ci`, `W_co`, and `W_cf` are learnable weight matrices for input gate, output gate, and forget gate, respectively.
- Decay factor: A learnable weight `decay_factor` controls the time-aware decay.
- Activation function: Supports both "tanh" and "relu" activation functions.
