import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Flatten,
    RepeatVector,
    LSTM,
    Dense,
    TimeDistributed,
    Reshape,
    ConvLSTM2D # For comparison or alternative decoder
)
from tensorflow.keras.models import Model

import sys
import os

# Get the absolute path to the 'root' directory
# This assumes 'main.py' is in 'root/example/'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the 'root' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now you can import from the 'tf' module
from tf.DA_ConvLSTM import DurationAwareConvLSTMCellTF
# 


# --- Duration-Aware ConvLSTM Autoencoder Model Creation ---
def create_duration_aware_convlstm_ae(
    input_frames_shape, # (window_size, height, width, channels)
    input_durations_shape, # (window_size,)
    filters_encoder=[60, 40],
    kernel_sizes_encoder=[(3,3), (3,3)],
    lstm_units_decoder=[40, 60],
    duration_cell_params=None # Dict for DurationAwareConvLSTMCellTF params
    ):
    """
    Creates a Duration-Aware ConvLSTM Autoencoder model.

    Args:
        input_frames_shape (tuple): Shape of the input frame sequences 
                                   (window_size, height, width, channels).
        input_durations_shape (tuple): Shape of the input duration sequences 
                                      (window_size,).
        filters_encoder (list): List of filter counts for encoder RNN layers.
        kernel_sizes_encoder (list): List of kernel sizes for encoder RNN layers.
        lstm_units_decoder (list): List of LSTM unit counts for decoder layers.
        duration_cell_params (dict, optional): Parameters for DurationAwareConvLSTMCellTF.
                                              Defaults to basic settings.

    Returns:
        tf.keras.models.Model: The compiled autoencoder model.
    """
    window_size, height, width, channels = input_frames_shape
    
    default_cell_params = {
        "duration_mode": "log",
        "batch_norm_durations": False,
        "duration_feature_dim": 16,
        "exp_damping_factor": 0.1
    }
    if duration_cell_params:
        default_cell_params.update(duration_cell_params)

    # --- Inputs ---
    frames_input = Input(shape=input_frames_shape, name="frames_input")
    durations_input = Input(shape=input_durations_shape, name="durations_input")

    # --- Encoder ---
    # The DurationAwareConvLSTMCellTF expects a tuple of (frame_tensor_t, duration_t)
    # The RNN layer handles unrolling over the time dimension (window_size).
    # We need to feed both frames and durations to the RNN layer.
    # The input to RNN will be a tuple of tensors: (frames_input, durations_input)
    # where frames_input is (batch, window_size, H, W, C)
    # and durations_input is (batch, window_size)
    
    x = (frames_input, durations_input) # Pack inputs for the RNN layer

    # Encoder Layer 1
    cell1_params = default_cell_params.copy()
    cell1 = DurationAwareConvLSTMCellTF(
        filters=filters_encoder[0], 
        kernel_size=kernel_sizes_encoder[0],
        name="da_convlstm_cell_1",
        **cell1_params
    )
    encoder_rnn1 = tf.keras.layers.RNN(cell1, return_sequences=True, name="encoder_rnn_1")
    # The RNN layer will iterate over the time axis (window_size),
    # passing (frames_input[:, t, ...], durations_input[:, t]) to cell1.call at each step t.
    encoded_sequence = encoder_rnn1(x) # Output shape: (batch, window_size, H, W, filters_encoder[0])

    # Encoder Layer 2
    cell2_params = default_cell_params.copy()
    cell2 = DurationAwareConvLSTMCellTF(
        filters=filters_encoder[1], 
        kernel_size=kernel_sizes_encoder[1],
        name="da_convlstm_cell_2",
        **cell2_params
    )
    encoder_rnn2 = tf.keras.layers.RNN(cell2, return_sequences=False, name="encoder_rnn_2") 
    # To pass the output of rnn1 (a sequence) and the original durations to rnn2,
    # we need to ensure rnn2 also receives durations for each step of its input sequence.
    encoded_latent = encoder_rnn2((encoded_sequence, durations_input)) # Output shape: (batch, H, W, filters_encoder[1])
    
    flattened_latent = Flatten(name="flatten_latent")(encoded_latent)

    # --- Bridge ---
    repeated_latent = RepeatVector(window_size, name="repeat_vector")(flattened_latent)

    # --- Decoder ---
    decoder_lstm1 = LSTM(
        lstm_units_decoder[0], activation='relu', return_sequences=True, name="decoder_lstm_1"
    )(repeated_latent)
    decoder_lstm2 = LSTM(
        lstm_units_decoder[1], activation='relu', return_sequences=True, name="decoder_lstm_2"
    )(decoder_lstm1)

    # Output layer to reconstruct the frames
    # The target is typically the input frames, possibly flattened per time step.
    # Output shape per time step should be height * width * channels
    output_dim = height * width * channels
    reconstructed_flat = TimeDistributed(
        Dense(output_dim, activation='sigmoid'), name="time_distributed_dense_output" # Sigmoid for normalized pixel values
    )(decoder_lstm2) # Output: (batch, window_size, H*W*C)

    # Optionally reshape to the original frame sequence shape if needed,
    # but often training is done on the flattened representation per timestep.
    # If target y is (batch, window_size, H, W, C), then add Reshape:
    reconstructed_frames = Reshape((window_size, height, width, channels), name="reshape_output")(reconstructed_flat)

    # --- Model Definition ---
    autoencoder = Model(inputs=[frames_input, durations_input], outputs=reconstructed_frames, name="duration_aware_convlstm_ae")
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

# --- Data Loader for NPZ files ---
def load_and_prepare_data_for_da_convlstm_ae(npz_filepath, window_size, step=1, grid_height=10, grid_width=8):
    """
    Loads frames and durations from an .npz file and prepares them for the 
    Duration-Aware ConvLSTM Autoencoder.

    Args:
        npz_filepath (str): Path to the .npz file.
        window_size (int): The number of time steps in each window/sequence.
        step (int, optional): The step size between consecutive windows. Defaults to 1.
        grid_height (int): Height of the sensor grid.
        grid_width (int): Width of the sensor grid.

    Returns:
        tuple: (X_frames_windowed, X_durations_windowed, y_target_windowed)
               Returns (None, None, None) if loading fails or no data.
    """
    if not os.path.exists(npz_filepath):
        print(f"Error: File not found at {npz_filepath}")
        return None, None, None
        
    try:
        data = np.load(npz_filepath)
        if 'frames' not in data or 'durations' not in data:
            print(f"Error: The file {npz_filepath} does not contain 'frames' or 'durations' arrays.")
            return None, None, None
        frames_array = data['frames']      # Expected shape: (N, H, W)
        durations_array = data['durations'] # Expected shape: (N,)
    except Exception as e:
        print(f"Error loading data from {npz_filepath}: {e}")
        return None, None, None

    if frames_array.ndim == 2: # If frames are (N, H*W)
        print(f"Original frames seem flattened (shape: {frames_array.shape}). Reshaping to (N, H, W).")
        frames_array = frames_array.reshape((-1, grid_height, grid_width))
    
    if frames_array.ndim != 3 or durations_array.ndim != 1 or frames_array.shape[0] != durations_array.shape[0]:
        print("Error: Frames or durations array has unexpected dimensions.")
        print(f"Frames shape: {frames_array.shape}, Durations shape: {durations_array.shape}")
        return None, None, None

    # Add channel dimension to frames: (N, H, W) -> (N, H, W, 1)
    frames_array_ch = np.expand_dims(frames_array, axis=-1)
    num_samples, h, w, c = frames_array_ch.shape
    
    if num_samples < window_size:
        print(f"Error: Not enough samples ({num_samples}) to create a window of size {window_size}.")
        return None, None, None

    X_frames_windowed = []
    X_durations_windowed = []
    
    for i in range(0, num_samples - window_size + 1, step):
        frame_seq = frames_array_ch[i : i + window_size]
        duration_seq = durations_array[i : i + window_size]
        X_frames_windowed.append(frame_seq)
        X_durations_windowed.append(duration_seq)
        
    if not X_frames_windowed:
        print("No windows created. Check data length and window parameters.")
        return None, None, None

    X_frames_windowed = np.array(X_frames_windowed)     # Shape: (num_windows, window_size, H, W, C)
    X_durations_windowed = np.array(X_durations_windowed) # Shape: (num_windows, window_size)
    
    # For autoencoder, target is the input itself (frames)
    # The model's final output is (batch, window_size, H, W, C)
    y_target_windowed = X_frames_windowed.copy() 
    
    print(f"Data loaded and prepared: ")
    print(f"  X_frames_windowed shape: {X_frames_windowed.shape}")
    print(f"  X_durations_windowed shape: {X_durations_windowed.shape}")
    print(f"  y_target_windowed shape: {y_target_windowed.shape}")
    
    return X_frames_windowed, X_durations_windowed, y_target_windowed


# --- Example Usage ---
if __name__ == "__main__":
    # --- 1. Generate Dummy NPZ data (mimicking sensor_log_processor.py output) ---
    print("--- Generating dummy NPZ data ---")
    dummy_num_frames = 200
    dummy_grid_height = 10
    dummy_grid_width = 8
    dummy_frames = np.random.randint(0, 2, size=(dummy_num_frames, dummy_grid_height, dummy_grid_width), dtype=np.uint8)
    dummy_durations = np.random.rand(dummy_num_frames).astype(np.float32) * 10 + 0.1 # Durations > 0
    dummy_npz_file = "dummy_processed_sensor_frames.npz"
    np.savez_compressed(dummy_npz_file, frames=dummy_frames, durations=dummy_durations)
    print(f"Dummy data saved to {dummy_npz_file}")

    # --- 2. Load and Prepare Data ---
    print("\n--- Loading and Preparing Data ---")
    WINDOW_SIZE = 30 # From user's notebook example
    STEP = 6         # From user's notebook example
    
    X_frames, X_durations, y_target = load_and_prepare_data_for_da_convlstm_ae(
        dummy_npz_file, 
        window_size=WINDOW_SIZE, 
        step=STEP,
        grid_height=dummy_grid_height,
        grid_width=dummy_grid_width
    )

    if X_frames is not None:
        print(f"Successfully loaded and windowed data.")
        
        # --- 3. Create the Model ---
        print("\n--- Creating Duration-Aware ConvLSTM Autoencoder Model ---")
        input_f_shape = (WINDOW_SIZE, dummy_grid_height, dummy_grid_width, 1) # (ws, h, w, c)
        input_d_shape = (WINDOW_SIZE,) # (ws,)

        # Example cell parameters (can be customized)
        cell_params = {
            "duration_mode": "exp",       # 'log' or 'exp'
            "batch_norm_durations": True, # True or False
            "duration_feature_dim": 8,    # Dimension for MLP hidden layer
            "exp_damping_factor": 0.05    # Damping for 'exp' mode
        }

        autoencoder_model = create_duration_aware_convlstm_ae(
            input_frames_shape=input_f_shape,
            input_durations_shape=input_d_shape,
            filters_encoder=[32, 16], # Reduced filters for dummy data
            kernel_sizes_encoder=[(3,3), (3,3)],
            lstm_units_decoder=[32, 32], # Reduced units for dummy data
            duration_cell_params=cell_params
        )
        autoencoder_model.summary()

        # --- 4. Train the Model (Dummy Training) ---
        print("\n--- Dummy Training the Model ---")
        if X_frames.shape[0] > 0: # Check if any windows were created
            history = autoencoder_model.fit(
                [X_frames, X_durations],  # Inputs as a list
                y_target,                 # Target
                epochs=3,                 # Few epochs for demonstration
                batch_size=16,            # Small batch size
                verbose=1
            )
            print("Dummy training complete.")
            print("Training history (loss):", history.history['loss'])

            # --- 5. Prediction Example (Optional) ---
            print("\n--- Prediction Example ---")
            if X_frames.shape[0] > 5:
                sample_frames_pred = X_frames[:5]
                sample_durations_pred = X_durations[:5]
                predictions = autoencoder_model.predict([sample_frames_pred, sample_durations_pred])
                print(f"Predictions shape: {predictions.shape}") # Should match y_target shape for the batch
            else:
                print("Not enough data for prediction example after windowing.")

        else:
            print("Skipping training as no data windows were generated.")
    else:
        print("Failed to load or prepare data. Skipping model creation and training.")

    # Clean up dummy file
    if os.path.exists(dummy_npz_file):
        os.remove(dummy_npz_file)
        print(f"\nCleaned up {dummy_npz_file}")
