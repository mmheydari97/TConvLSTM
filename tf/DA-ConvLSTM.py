import tensorflow as tf

class DurationAwareConvLSTMCell(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, data_format='channels_last', duration_feature_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.data_format = data_format
        self.state_size = [tf.TensorShape([None, None, filters]), tf.TensorShape([None, None, filters])] # h, c

        self.conv_gates = tf.keras.layers.Conv2D(
            filters=4 * filters, kernel_size=kernel_size, padding='same', data_format=data_format)

        self.duration_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(duration_feature_dim, activation='relu'),
            tf.keras.layers.Dense(1) # Output a single factor or `filters` for channel-wise
        ])

    def build(self, input_shape): # input_shape is for x_t
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        # input_shape[0] is for the frame tensor
        # input_shape[1] is for the duration scalar, we don't use it for conv kernel building.
        if isinstance(input_shape, list): # If inputs=(frames, durations)
            frame_shape = input_shape[0]
        else: # Should not happen if used with RNN layer correctly
            frame_shape = input_shape

        input_dim = frame_shape[channel_axis]
        # self.conv_gates (defined in __init__) will be built implicitly on first call

    def call(self, inputs, states, training=None):
        # `inputs` will be (input_tensor_t, duration_t)
        # `states` will be [h_prev, c_prev]
        input_tensor_t, duration_t = inputs
        h_prev, c_prev = states

        combined = tf.concat([input_tensor_t, h_prev], axis=-1 if self.data_format == 'channels_last' else 1)
        gates = self.conv_gates(combined)

        i, f, g, o = tf.split(gates, num_or_size_splits=4, axis=-1 if self.data_format == 'channels_last' else 1)

        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        g_tilde = tf.tanh(g) # Candidate cell state
        o = tf.sigmoid(o)

        # Process duration
        duration_t_reshaped = tf.expand_dims(tf.cast(duration_t, tf.float32), -1) # [B, 1]
        # duration_factor = 1.0 + tf.nn.relu(self.duration_mlp(duration_t_reshaped)) # [B, 1]
        # For broadcasting:
        # duration_factor = tf.reshape(duration_factor, [-1, 1, 1, 1]) # For channels_last: B,1,1,1
        
        # Alternative simple log based:
        duration_factor = tf.math.log(tf.cast(duration_t, tf.float32) + 1e-6) + 1.0 # [B]
        # Reshape for broadcasting: (B, 1, 1, 1) if channels_last; (B, 1, 1, 1) actually works for both if channels is not axis 0
        if self.data_format == 'channels_last':
            shape = [-1, 1, 1, 1]
        else: # channels_first
            shape = [-1, 1, 1, 1] # This still works because factor applies to all channels
                                   # Or use tf.expand_dims multiple times. This is safer for channels_first:
                                   # shape = [-1, 1, 1, 1] and then tf.transpose if needed, or more simply
            # duration_factor = tf.reshape(duration_factor, [-1, 1, 1, 1]) # B, 1, H, W for scalar multiplier.

        # Let's assume duration_mlp outputs a single value to scale g_tilde
        # And ensure it's positive and doesn't explode
        duration_scalar_effect = self.duration_mlp(duration_t_reshaped) # [B, 1]
        # This factor scales the entire feature map g_tilde
        _duration_factor = tf.exp(duration_scalar_effect * 0.1) # [B,1], apply some damping if needed
        if self.data_format == 'channels_last': # B, H, W, C
            _duration_factor_reshaped = tf.reshape(_duration_factor, [-1, 1, 1, 1])
        else: # B, C, H, W
            _duration_factor_reshaped = tf.reshape(_duration_factor, [-1, 1, 1, 1]) # Will broadcast over C, H, W

        g_modified = g_tilde * _duration_factor_reshaped

        c_next = f * c_prev + i * g_modified
        h_next = o * tf.tanh(c_next)

        return h_next, [h_next, c_next]

# To use it:
# cell = DurationAwareConvLSTMCell(filters=64, kernel_size=(3,3))
# rnn_layer = tf.keras.layers.RNN(cell, return_sequences=True)
#
# # Input: sequence of frames and sequence of durations
# # frames_input shape: (batch, seq_len, H, W, C)
# # durations_input shape: (batch, seq_len)
# outputs = rnn_layer((frames_input, durations_input))
