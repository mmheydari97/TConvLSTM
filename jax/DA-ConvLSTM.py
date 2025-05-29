import jax
import jax.numpy as jnp
import flax.linen as nn

class DurationAwareConvLSTMCellFlax(nn.Module):
    hidden_features: int
    kernel_size: tuple[int, int]
    duration_mode: str = "log"  # "exp" or "log"
    batch_norm_durations: bool = False
    duration_feature_dim: int = 16
    exp_damping_factor: float = 0.1
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, carry, inputs_t, training: bool):
        # carry: (h_prev, c_prev)
        # inputs_t: (x_t, duration_t)
        # x_t: (H, W, C_in) - Assuming per-sample, vmap handles batch
        # duration_t: scalar (per-sample)
        h_prev, c_prev = carry
        x_t, d_t = inputs_t

        # Conv operates on (H, W, C_in + C_hidden)
        combined = jnp.concatenate([x_t, h_prev], axis=-1)
        gates = nn.Conv(features=4 * self.hidden_features,
                        kernel_size=self.kernel_size,
                        padding='SAME',
                        name="gates_conv")(combined)
        
        i_val, f_val, g_val, o_val = jnp.split(gates, 4, axis=-1)

        i = nn.sigmoid(i_val)
        f = nn.sigmoid(f_val)
        g = jnp.tanh(g_val) # Candidate cell state
        o = nn.sigmoid(o_val)

        # --- Duration Processing ---
        # d_t is per-sample scalar. If vmap is used, batch_norm_durations needs pmean/pstd
        # or it normalizes per-sample (which is trivial for a scalar).
        # Assuming d_t here is the batched version [B] if this cell is vmap'd or inside Scan over sequence.
        # For this example, let's assume d_t is a batch of durations [B] passed to a batched call.
        
        d_val = d_t.astype(jnp.float32) # Shape: [B] if batched

        if self.batch_norm_durations:
            # This normalizes across the batch dimension.
            # In JAX, if this cell is used within a `vmap` for batching,
            # `d_val` would be a single scalar per vmap call.
            # To normalize across a true batch, this logic should be outside a per-sample vmap,
            # or use `jax.lax.pmean`/`pstd` if distributed.
            # For a simple batched input to a non-vmapped layer:
            mean_d = jnp.mean(d_val, axis=0, keepdims=True)
            std_d = jnp.std(d_val, axis=0, keepdims=True)
            d_val = (d_val - mean_d) / (std_d + self.epsilon)

        # MLP input shape [B, 1]
        mlp_input = jnp.expand_dims(d_val, axis=-1)
        
        scalar_effect = nn.Dense(features=self.duration_feature_dim, name="duration_mlp_dense1")(mlp_input)
        scalar_effect = nn.relu(scalar_effect)
        scalar_effect = nn.Dense(features=1, name="duration_mlp_dense2")(scalar_effect) # Shape [B, 1]

        if self.duration_mode == "log":
            duration_factor = 1.0 + nn.relu(scalar_effect)
        elif self.duration_mode == "exp":
            duration_factor = jnp.exp(scalar_effect * self.exp_damping_factor)
        else:
            raise ValueError(f"Unknown duration_mode: {self.duration_mode}")
        
        # Reshape factor for broadcasting: [B, 1, 1, 1]
        # scalar_effect and duration_factor are [B,1]. We need [B,1,1,1] for g [B,H,W,C]
        factor_reshaped = jnp.reshape(duration_factor, (duration_factor.shape[0], 1, 1, 1))
        
        g_modified = g * factor_reshaped
        # --- End Duration Processing ---

        c_next = f * c_prev + i * g_modified
        h_next = o * jnp.tanh(c_next)

        return (h_next, c_next), h_next # scan returns (new_carry, y_t)

    @staticmethod
    def init_carry(batch_size, hidden_features, image_dims, key):
        # image_dims: (H, W)
        # key for potential initializations if needed, not strictly for zeros
        h = jnp.zeros((batch_size, image_dims[0], image_dims[1], hidden_features))
        c = jnp.zeros((batch_size, image_dims[0], image_dims[1], hidden_features))
        return (h, c)

# To use it in a layer that processes sequences (using nn.scan)
class DurationConvLSTMFlaxLayer(nn.Module):
    hidden_features: int
    kernel_size: tuple[int, int]
    duration_mode: str = "log"
    batch_norm_durations: bool = False
    # ... other params for cell ...

    @nn.compact
    def __call__(self, frames_sequence, durations_sequence, training: bool):
        # frames_sequence: (B, SeqLen, H, W, C_in)
        # durations_sequence: (B, SeqLen)
        
        # Transpose for scan: (SeqLen, B, ...)
        frames_scan = jnp.transpose(frames_sequence, (1, 0, 2, 3, 4))
        durations_scan = jnp.transpose(durations_sequence, (1, 0))
        
        scan_inputs = (frames_scan, durations_scan) # (SeqLen, B, H, W, C_in) and (SeqLen, B)

        cell = DurationAwareConvLSTMCellFlax(
            hidden_features=self.hidden_features,
            kernel_size=self.kernel_size,
            duration_mode=self.duration_mode,
            batch_norm_durations=self.batch_norm_durations,
            # ... pass other params ...
            name="duration_conv_lstm_cell" # Name the cell sub-module
        )
        
        batch_size = frames_sequence.shape[0]
        image_dims = (frames_sequence.shape[2], frames_sequence.shape[3])
        
        # For nn.scan, the 'training' flag needs to be passed into the scan function's body if cell uses it.
        # Flax's scan is tricky with how it passes static arguments like 'training' to the scanned function.
        # One way is to use a partial function or ensure 'training' is part of the input/carry if it changes per step.
        # For this cell, 'training' isn't directly used by the cell's Flax modules (like nn.Dense)
        # unless a Flax nn.BatchNorm was used. Our manual BN for durations doesn't use a 'training' flag.
        
        ScannedLSTM = nn.scan(
            type(cell), # Pass the cell type
            variable_broadcast="params", # Standard for Flax scans
            split_rngs={"params": False}, # Standard
            in_axes=0, # Scan over SeqLen axis of inputs
            out_axes=0 # Output has SeqLen axis first
        )
        
        # initial_carry must be created for the actual batch size
        # key for init is not used by current static init_carry
        # If cell was vmapped, training would be passed to vmap.
        # Here, cell.apply is called by scan.
        initial_carry_val = DurationAwareConvLSTMCellFlax.init_carry(
            batch_size, self.hidden_features, image_dims, jax.random.key(0) # dummy key
        )

        # The 'training' argument needs to be correctly plumbed if cell had modules like nn.BatchNorm
        # that behave differently during training/inference.
        # Our manual duration BN is independent of a global `training` flag in this impl.
        # If `cell` itself needed `training`, you'd use:
        # `lambda carry, x: cell(carry, x, training=training)`
        # However, nn.scan's body function doesn't directly accept extra static args like that.
        # A common way is to make cell a sub-module and call it.
        
        final_carry, outputs_h_scanned = ScannedLSTM(methods={'__call__': lambda c, i: cell(c, i, training=training)})(initial_carry_val, scan_inputs)

        # Transpose back: (B, SeqLen, H, W, C_hidden)
        outputs_h = jnp.transpose(outputs_h_scanned, (1, 0, 2, 3, 4))
        
        return outputs_h, final_carry
    
    
# To use for a sequence (outside the Module if using scan directly, or within a higher-level Module):
# cell = DurationAwareConvLSTMCell(hidden_features=64, kernel_size=(3,3))
# initial_h = jnp.zeros((batch_size, H, W, hidden_features)) # Assuming batching handled by vmap or scan over batch
# initial_c = jnp.zeros((batch_size, H, W, hidden_features))
# initial_state = (initial_h, initial_c)
#
# # inputs_frames: (seq_len, H, W, C_in)
# # inputs_durations: (seq_len,)
# # Combine them for scan:
# scan_inputs = (inputs_frames, inputs_durations)
#
# final_state, outputs_h = nn.scan(DurationAwareConvLSTMCell.apply, # Or cell.apply if cell is instantiated
#                                  variable_broadcast="params", # Flax specific for params
#                                  split_rngs={"params": False}, # Flax specific
#                                  in_axes=0, # Scan over sequence dim
#                                  out_axes=0
#                                 )(initial_state, scan_inputs)
# Note: Batching with vmap or handling inside scan might be needed for `initial_state` and `inputs`.
# A common pattern is to wrap the cell in another nn.Module that performs the scan.
