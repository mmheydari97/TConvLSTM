import numpy as np
import tensorflow as tf


class TConvLSTMCell(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, activation, frame_size):
        super(TConvLSTMCell, self).__init__()
        
        if activation == "tanh":
            self.activation = tf.nn.tanh
        elif activation == "relu":
            self.activation = tf.nn.relu
        
        self.conv = tf.keras.layers.Conv2D(
            filters=4 * filters,
            kernel_size=kernel_size,
            padding=padding)
        
        self.W_ci = self.add_weight(
            shape=(filters, *frame_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True)
        
        self.W_co = self.add_weight(
            shape=(filters, *frame_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True)
        
        self.W_cf = self.add_weight(
            shape=(filters, *frame_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True)
        
        self.decay_factor = tf.Variable(0.0)

    def call(self, X, duration, H_prev, C_prev):
        conv_output = self.conv(tf.concat([X, H_prev], axis=-1))
        i_conv, f_conv, C_conv, o_conv = tf.split(conv_output, num_or_size_splits=4, axis=-1)

        input_gate = tf.sigmoid(i_conv + self.W_ci * C_prev)
        decay_factor = tf.exp(-self.decay_factor * duration)
        forget_gate = tf.sigmoid(f_conv + self.W_cf * C_prev) * decay_factor
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)
        output_gate = tf.sigmoid(o_conv + self.W_co * C)
        H = output_gate * self.activation(C)

        return H, C


class TConvLSTM(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, activation, frame_size, return_sequence=False):
        super(TConvLSTM, self).__init__()
        self.filters = filters
        self.return_sequence = return_sequence
        
        self.cell = TConvLSTMCell(filters, kernel_size, padding, activation, frame_size)

    def call(self, X, durations):
        print(X.shape)
        seq_len, height, width, channels = X.shape[1:]
        
        output = tf.zeros((seq_len, height, width, self.filters))
        
        H = tf.zeros((height, width, self.filters))
        
        C = tf.zeros((height, width, self.filters))

        for t, duration in enumerate(durations):
            H, C = self.cell(X[:, t], duration, H, C)
            output[t, ...].assign(H)
        
        if not self.return_sequence:
            output = tf.squeeze(output[-1, ...], axis=0)
        
        return output

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)

    # Define the test parameters
    batch_size = 2
    seq_len = 4
    height = 32
    width = 32
    channels = 3
    filters = 64
    kernel_size = (3, 3)
    padding = 'same'
    activation = 'tanh'
    frame_size = (height, width)
    return_sequence = False

    # Create random input tensors
    X = tf.random.normal((batch_size, seq_len, height, width, channels))
    durations = tf.random.uniform((batch_size, seq_len), minval=0, maxval=1)

    # Create the TConvLSTM layer
    tconv_lstm = TConvLSTM(filters, kernel_size, padding, activation, frame_size, return_sequence)

    # Pass the inputs through the layer
    output = tconv_lstm(X, durations)

    # Print the output shape
    print("Output shape:", output.shape)