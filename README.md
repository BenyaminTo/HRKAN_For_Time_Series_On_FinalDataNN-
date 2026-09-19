# HRKAN-LSTM Time Series Forecasting

A PyTorch implementation of a hybrid **Higher-order Reduced Kolmogorov-Arnold Network (HRKAN)** + **LSTM** model for time series forecasting. The model uses a custom HRKAN layer (based on B-spline-like basis functions and a `Conv2d` projection) to extract nonlinear features from short sub-sequences, then feeds the resulting embeddings into an LSTM to forecast future values.

## Overview

The pipeline:

1. Loads paired input/output time series (`P`, `Y`) from an Excel file.
2. Builds sliding-window samples: a `context_len`-long window of `P` is used to predict the following `pred_len`-long window of `Y`.
3. Splits each context window into smaller chunks of size `seq_step` and passes each chunk through an **HRKAN layer**, which learns nonlinear basis-function features (inspired by Kolmogorov-Arnold Networks) instead of standard fixed activation functions.
4. Feeds the sequence of HRKAN embeddings into an **LSTM**, and uses the final hidden state to predict the forecast horizon via a linear output layer.
5. Trains the model, tracks MAE/MSE and timing per epoch, and plots training curves and predictions against ground truth.

## Model Architecture

### `HRKANLayer`
A custom nonlinear layer that:
- Defines a set of grid intervals (`phase_low`, `phase_height`) over the input range `[imin, imax]`, controlled by grid size `g` and spline order `k`.
- Builds bounded, triangular-like basis functions from the input using ReLU.
- Raises these basis functions to a configurable power (`order`, "Higher-Order ReLU") to increase nonlinearity.
- Projects the resulting basis-function tensor to `output_size` features using a `Conv2d` layer.

> Note: `HRKANLayer` is defined twice in the notebook/script (cells 6 and 7). The second definition is functionally equivalent but adds input validation (`imax > imin`) and an explicit `.float()` cast, and is the one actually used to build the model, since it overwrites the first.

### `RecurrentHRKANTS`
The full forecasting model:
1. Splits the input context window of length `context_len` into `num_steps = context_len / seq_step` chunks.
2. Passes all chunks through a shared `HRKANLayer` to get a sequence of embeddings.
3. Feeds the embedding sequence into a multi-layer LSTM (optionally bidirectional).
4. Applies a linear layer to the LSTM's final hidden state to output `pred_len` forecasted values.

## Requirements

```
numpy
pandas
matplotlib
torch
openpyxl   # required by pandas to read .xlsx files
tabulate   # required for DataFrame.to_markdown()
```

Install with:

```bash
pip install numpy pandas matplotlib torch openpyxl tabulate
```

## Data

The script expects an Excel file with (at least) the following columns:

| Column  | Description                     |
|---------|----------------------------------|
| `P`     | Training input series            |
| `Y`     | Training target series           |
| `Ptest` | Test input series                |
| `Ytest` | Test target series                |

Rows with missing values in any of these columns are dropped.

**Before running:** update the `excel_path` variable near the top of the script to point to your local copy of the dataset:

```python
excel_path = "/FinalDataNN.xlsx"
```

## Hyperparameters

| Name            | Description                                    | Default |
|-----------------|-------------------------------------------------|---------|
| `K_S`           | Spline order (`k`)                              | 3       |
| `G_S`           | Grid size (`g`)                                 | 3       |
| `IMIN` / `IMAX` | Input range for the HRKAN basis functions        | 0.0 / 1.0 |
| `Order`         | Nonlinearization power of the Higher-Order ReLU  | 2       |
| `Batch_Size`    | Training batch size                              | 32      |
| `Context_Len`   | Length of the input context window               | 120     |
| `Pred_Len`      | Forecast horizon length                          | 20      |
| `H_Dims`        | HRKAN output dim / LSTM hidden size              | 32      |
| `N_Epochs`      | Number of training epochs                        | 3000    |
| `Learning_Rate` | AdamW learning rate                               | 3e-4    |
| `Seq_Step`      | Chunk size the context window is split into      | 5       |
| `Num_Layers`    | Number of stacked LSTM layers                     | 2       |
| `Bidirectional` | Whether the LSTM is bidirectional                 | False   |

`Context_Len` must be evenly divisible by `Seq_Step`.

## Usage

Run the script directly:

```bash
python HRKAN_LSTM_TS_for_FinalDataNN.py
```

This will:
1. Load and clean the dataset, and plot the raw `P`/`Y` series.
2. Build sliding-window train/test datasets and data loaders.
3. Build the `RecurrentHRKANTS` model and move it to GPU if available.
4. Train for `N_Epochs` epochs, printing loss, MAE, MSE, and timing information each epoch.
5. Compute final train/test MAE and MSE, estimate a speed-up factor against a dummy "physics simulation" baseline, and print a Markdown summary table.
6. Plot MAE/MSE training curves and the full test-set prediction vs. ground truth.

## Outputs

- **Console logs**: per-epoch loss, MAE, MSE, and timing; a final Markdown results table (train/test MAE & MSE, speed-up factor, total training time, average inference time, average per-iteration training time, parameter count, model structure).
- **Plots**:
  - Raw `P` and `Y` time series.
  - MAE per epoch (train vs. test).
  - MSE per epoch (train vs. test).
  - Full test-set predictions vs. actual values.

## Notes

- The device is selected automatically (`cuda` if available, otherwise `cpu`).
- The loss function is **MAE (L1Loss)**; MSE/RMSE are also tracked for evaluation.
- The "speed-up" metric compares total time spent in a placeholder `physics_simulation` function (a `time.sleep` proportional to batch size) against the model's average inference time, as a rough proxy for how much faster the learned model is than a simulation-based baseline.
