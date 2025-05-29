import jax
import jax.numpy as jnp
import flax.linen as nn # Or haiku as hk

class DurationAwareConvLSTMCell(nn.Module):
    hidden_features: int
    kernel_size: tuple[int, int]
    duration_feature_dim: int = 16

    @nn.compact
    def __call__(self, carry, inputs):
        # carry: (h_prev, c_prev)
        # inputs: (x_t, duration_t)
        h_prev, c_prev = carry
        x_t, d_t = inputs # x_t: (H, W, C_in), d_t: scalar

        # Assuming channels_last for JAX/Flax Conv
        combined = jnp.concatenate([x_t, h_prev], axis=-1)
        gates = nn.Conv(features=4 * self.hidden_features, kernel_size=self.kernel_size, padding='SAME')(combined)
        
        i_val, f_val, g_val, o_val = jnp.split(gates, 4, axis=-1)

        i = nn.sigmoid(i_val)
        f = nn.sigmoid(f_val)
        g_tilde = jnp.tanh(g_val)
        o = nn.sigmoid(o_val)

        # Process duration (d_t is per-sample, needs to be [1] or [])
        d_t_reshaped = d_t.astype(jnp.float32).reshape((1,)) # For MLP expecting [batch, features]

        duration_effect = nn.Dense(features=self.duration_feature_dim, name="duration_mlp_1")(d_t_reshaped)
        duration_effect = nn.relu(duration_effect)
        duration_scalar = nn.Dense(features=1, name="duration_mlp_2")(duration_effect) # outputs [1]
        
        # duration_factor = jnp.exp(duration_scalar[0] * 0.1) # Scalar factor
        # A simpler log based:
        duration_factor = jnp.log(d_t.astype(jnp.float32) + 1e-6) + 1.0


        g_modified = g_tilde * duration_factor # Broadcasting scalar

        c_next = f * c_prev + i * g_modified
        h_next = o * jnp.tanh(c_next)

        return (h_next, c_next), h_next # scan returns (new_carry, y_t)

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
