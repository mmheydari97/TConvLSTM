# Duration-Aware ConvLSTM

A unified implementation of a **Duration-Aware Convolutional LSTM Cell** across **PyTorch**, **TensorFlow (Keras)**, and **JAX (Flax)** frameworks. This model integrates temporal duration information into the ConvLSTM cell update, making it suitable for spatiotemporal modeling tasks where variable-duration events play a significant role.

---

## 📌 Motivation

In standard ConvLSTM cells, the time intervals between frames are assumed to be uniform. However, many real-world sequences (e.g., clinical time series, surveillance video, event-based vision) contain *non-uniform durations*. This module modifies the classic ConvLSTM to:

* Accept a scalar **duration input** at each timestep.
* Modulate the candidate cell state using a learned function of the duration.
* Allow *temporal irregularity* to influence state updates.

---

## 📦 Features

* ✅ Implemented in **PyTorch**, **TensorFlow (Keras)**, and **JAX (Flax)**.
* 📐 Flexible architecture for kernel size, hidden channels, and MLP structure.
* ⏱️ Multiple strategies for modeling duration impact (linear, log, learned scalar).
* 🧠 Ready for integration with sequence models (e.g., `RNN`, `scan`, looped inference).

---

## 🔧 Installation

To run the code in each framework:

### PyTorch

```bash
pip install torch
```

### TensorFlow

```bash
pip install tensorflow
```

### JAX (Flax or Haiku)

```bash
pip install flax optax jax jaxlib
```

> For GPU support in JAX, refer to: [https://github.com/google/jax#installation](https://github.com/google/jax#installation)

---

## 🧠 Architecture Overview

Each Duration-Aware ConvLSTM cell consists of:

* Standard ConvLSTM gates: input (i), forget (f), output (o), and candidate cell state (g̃).
* A **duration-aware MLP** that transforms a scalar duration into a multiplicative factor applied to `g̃`.
* The new cell state is computed as:

```
c_t = f ⊙ c_{t-1} + i ⊙ (g̃ × duration_factor)
h_t = o ⊙ tanh(c_t)
```

---

## 🧪 Usage

### PyTorch Example

```python
rnn_cell = DurationAwareConvLSTMCell(input_dim=3, hidden_dim=64, kernel_size=(3,3))
h, c = rnn_cell.init_hidden(batch_size=4, image_size=(64, 64))

for frame, duration in zip(sequence_frames, sequence_durations):
    h, c = rnn_cell(frame, duration, (h, c))
```

### TensorFlow Example

```python
cell = DurationAwareConvLSTMCell(filters=64, kernel_size=(3,3))
rnn_layer = tf.keras.layers.RNN(cell, return_sequences=True)

output = rnn_layer((frames_input, durations_input))  # Sequence input
```

### JAX (Flax) Example

```python
cell = DurationAwareConvLSTMCell(hidden_features=64, kernel_size=(3,3))
initial_state = (h0, c0)

def scan_fn(carry, inputs):
    return cell().apply({'params': params}, carry, inputs)

final_state, outputs = jax.lax.scan(scan_fn, initial_state, (frames_seq, durations_seq))
```

---

## 🔬 Duration Scaling Strategies

| Strategy         | Description                                 |
| ---------------- | ------------------------------------------- |
| Learned MLP      | Transforms duration → scaling vector/factor |
| Log-based        | Uses `log(duration + ε) + 1.0`              |
| Exponential Damp | Uses `exp(scalar × 0.1)` to control growth  |

These strategies can be toggled or customized based on your task sensitivity to time variance.

---

## 🧱 Applications

* Clinical time-series forecasting (e.g., irregular vitals)
* Event-based video summarization
* Spatiotemporal forecasting with non-uniform frame rates
* Motion prediction in robotics with variable time steps

---

