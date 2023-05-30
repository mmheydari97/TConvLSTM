import numpy as np
from tensorflow.keras.layers import Input, ConvLSTM2D, RepeatVector, Flatten, LSTM, TimeDistributed, Dense, Concatenate
from tensorflow.keras.models import Model
from time_conv_lstm import TConvLSTM


# def TConvLSTMAE(inputs, depth=2, windows_size=20,  ):
#     if depth == 1:
#         model.add(ConvLSTM2D(filters=100, kernel_size=(1,5), activation='relu',return_sequences = False, input_shape=(X.shape[1],1,X.shape[3], 1)))
#         model.add(Flatten())
#         model.add(RepeatVector(windows_size))
#         model.add(LSTM(60, activation='relu', return_sequences=True))
#         model.add(TimeDistributed(Dense(60, activation='relu')))
#         model.add(TimeDistributed(Dense(35))) 
#         model.summary()
#         model.compile(optimizer='adam', loss='mse')
#         history = model.fit(X,  X.reshape((-1, X_train.shape[1],X_train.shape[3])), 
#                             batch_size=batch_size, epochs=num_epochs,verbose=1).history       
#     else:
#         model = Sequential()
#         model.add(ConvLSTM2D(filters=60, kernel_size=(3,3), activation='relu',return_sequences = True,
#                              input_shape=(X.shape[1],X.shape[2],X.shape[3], 1)))
#         model.add(ConvLSTM2D(filters=40, kernel_size=(3,3), activation='relu',return_sequences = False))
#         model.add(Flatten())
#         model.add(RepeatVector(windows_size))
#         model.add(LSTM(40, activation='relu', return_sequences=True))
#         model.add(LSTM(60, activation='relu', return_sequences=True))
#         model.add(TimeDistributed(Dense(80, activation='relu')))
#         model.summary()
#         model.compile(optimizer='adam', loss='mse')
#         history = model.fit(X,  X.reshape((-1, X.shape[1], X.shape[2]*X.shape[3])), 
#                             batch_size=batch_size, epochs=num_epochs,verbose=1).history
#     return model


# class TConLSTMAE:
#     def __init__(self, frame_size, windows_size, input_channel):
#         self.frame_size = frame_size
#         self.windows_size = windows_size
#         self.input_channel = input_channel

#         self.model = self.build_model()

#     def build_model(self):
#         input_X = Input(shape=(self.windows_size, *self.frame_size, self.input_channel))
#         input_durations = Input(shape=(self.windows_size,))

#         convlstm1 = TConvLSTM(60, (3, 3), 'same', 'relu', self.frame_size, return_sequence=True)(input_X, input_durations)
#         convlstm2 = TConvLSTM(40, (3, 3), 'same', 'relu', self.frame_size, return_sequence=False)(convlstm1, input_durations)

#         last_item = convlstm1[:, -1]
#         skip_connection = Concatenate(axis=-1)([last_item, convlstm2])
#         flattened = Flatten()(skip_connection)

#         repeat_vector = RepeatVector(self.windows_size)(flattened)
#         lstm1 = LSTM(40, activation='relu', return_sequences=True)(repeat_vector)
#         lstm2 = LSTM(60, activation='relu', return_sequences=True)(lstm1)
#         output = TimeDistributed(Dense(80, activation='relu'))(lstm2)

#         model = Model(inputs=[input_X, input_durations], outputs=output)
#         model.compile(optimizer='adam', loss='mse')
#         return model

#     def fit(self, X, durations, batch_size, num_epochs):
#         history = self.model.fit([X, durations], X.reshape((-1, X.shape[2], X.shape[3]*X.shape[4])),
#                                  batch_size=batch_size, epochs=num_epochs, verbose=1).history
#         return history

#     def predict(self, X, durations):
#         return self.model.predict([X, durations])


if __name__ == "__main__":
    # Generate random input tensor and durations
    batch_size = 4
    sequence_len = 20
    frame_size = [32, 32]
    X = np.random.rand(100*batch_size, sequence_len, frame_size[0], frame_size[1], 3)
    durations = np.random.rand(batch_size, sequence_len)

    # Train the model
    model= TConLSTMAE(frame_size=frame_size, windows_size=sequence_len, input_channel=3)
    model.fit(X, durations, batch_size, 5)

    # Test the model on a random input
    test_X = np.random.rand(batch_size, sequence_len, *frame_size, 1)
    test_durations = np.random.rand(batch_size, sequence_len)
    predicted_X = model.predict(test_X, test_durations)

    print("Input shape:", test_X.shape)
    print("Output shape:", predicted_X.shape)