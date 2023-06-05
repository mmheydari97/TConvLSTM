import tensorflow as tf

# Define the input shape
height, width, channels = 32, 32, 3

# Define the model architecture
class TemporalAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[1][-1], 1),
                                 initializer='uniform',
                                 trainable=True)
        super(TemporalAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        frames, durations = inputs
        attention_weights = tf.nn.softmax(tf.matmul(durations, self.W), axis=1)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        attended_frames = frames * attention_weights
        return attended_frames

input_frames = tf.keras.Input(shape=(None, height, width, channels))
input_durations = tf.keras.Input(shape=(None, 1))

attended_frames = TemporalAttentionLayer()([input_frames, input_durations])
convlstm_output = tf.keras.layers.ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(attended_frames)

model = tf.keras.Model(inputs=[input_frames, input_durations], outputs=convlstm_output)

# Define the input data
frames = tf.random.normal((1, 10, height, width, channels))
durations = tf.random.normal((1, 10, 1))

# Test the model
output = model([frames, durations])
print(output.shape)
