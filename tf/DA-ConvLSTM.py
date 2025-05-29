import tensorflow as tf

class DurationAwareConvLSTMCellTF(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, data_format='channels_last',
                 duration_mode="log", batch_norm_durations=False,
                 duration_feature_dim=16, exp_damping_factor=0.1, epsilon=1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.data_format = data_format
        self.state_size = [tf.TensorShape([None, None, filters]), tf.TensorShape([None, None, filters])] # h, c
        self.epsilon = epsilon

        # Duration processing settings
        if duration_mode not in ["log", "exp"]:
            raise ValueError("duration_mode must be 'log' or 'exp'")
        self.duration_mode = duration_mode
        self.batch_norm_durations = batch_norm_durations
        self.exp_damping_factor = exp_damping_factor

        self.conv_gates = tf.keras.layers.Conv2D(
            filters=4 * filters, kernel_size=self.kernel_size, padding='same', data_format=data_format)

        self.duration_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(duration_feature_dim, activation='relu', name="duration_mlp_dense1"),
            tf.keras.layers.Dense(1, name="duration_mlp_dense2") # Output a single scalar effect
        ], name="duration_mlp")
        self.channel_axis = -1 if data_format == 'channels_last' else 1


    def call(self, inputs, states, training=None): # `training` flag from Keras
        # `inputs` will be (input_tensor_t, duration_t)
        # `states` will be [h_prev, c_prev]
        input_tensor_t, duration_t = inputs
        h_prev, c_prev = states

        combined = tf.concat([input_tensor_t, h_prev], axis=self.channel_axis)
        gates = self.conv_gates(combined)

        i, f, g_tilde, o = tf.split(gates, num_or_size_splits=4, axis=self.channel_axis)

        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        g = tf.tanh(g_tilde) # Candidate cell state
        o = tf.sigmoid(o)

        # --- Duration Processing ---
        d_val = tf.cast(duration_t, tf.float32) # Shape: [B]

        if self.batch_norm_durations:
            # Normalize across the batch for the current timestep
            # Note: In TF, mean/std over batch in `call` is fine for this custom logic.
            # For standard BN layers, `training` flag controls use of running stats.
            # Here, we always use batch stats if flag is true.
            mean_d = tf.reduce_mean(d_val, axis=0, keepdims=True)
            std_d = tf.math.reduce_std(d_val, axis=0, keepdims=True)
            d_val = (d_val - mean_d) / (std_d + self.epsilon)
        
        mlp_input = tf.expand_dims(d_val, axis=-1) # Shape: [B, 1]
        scalar_effect = self.duration_mlp(mlp_input, training=training) # Shape: [B, 1]

        if self.duration_mode == "log":
            duration_factor = 1.0 + tf.nn.relu(scalar_effect)
        elif self.duration_mode == "exp":
            duration_factor = tf.exp(scalar_effect * self.exp_damping_factor)

        # Reshape factor for broadcasting: e.g., [B, 1, 1, 1]
        # Get rank of g for robust reshaping
        g_shape = tf.shape(g)
        if self.data_format == 'channels_last': # B, H, W, C
            factor_reshaped = tf.reshape(duration_factor, [g_shape[0], 1, 1, 1])
        else: # B, C, H, W
            factor_reshaped = tf.reshape(duration_factor, [g_shape[0], 1, 1, 1]) # Will broadcast fine

        g_modified = g * factor_reshaped
        # --- End Duration Processing ---

        c_next = f * c_prev + i * g_modified
        h_next = o * tf.tanh(c_next)

        return h_next, [h_next, c_next]


# To use it with Keras RNN layer:
# cell = DurationAwareConvLSTMCellTF(filters=64, kernel_size=(3,3), duration_mode="exp", batch_norm_durations=True)
# rnn_layer = tf.keras.layers.RNN(cell, return_sequences=True)
#
# # Input: sequence of frames and sequence of durations
# frames_input = tf.keras.Input(shape=(None, H, W, C_in), name="frames") # B, Seq, H, W, C
# durations_input = tf.keras.Input(shape=(None,), name="durations")      # B, Seq
# outputs = rnn_layer((frames_input, durations_input))
# model = tf.keras.Model(inputs=[frames_input, durations_input], outputs=outputs)
