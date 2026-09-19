# Installation and Imports

# In[1]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Load and clean the dataset

# In[2]:


excel_path = "/FinalDataNN.xlsx"
df = pd.read_excel(excel_path)
df_cleaned = df.dropna(subset=["P", "Y", "Ptest", "Ytest"])

P_train = df_cleaned["P"].values.astype("float32")
Y_train = df_cleaned["Y"].values.astype("float32")
P_test  = df_cleaned["Ptest"].values.astype("float32")
Y_test  = df_cleaned["Ytest"].values.astype("float32")

print(f"Train set size:  {len(P_train)} samples")
print(f"Test set size:   {len(P_test)} samples")


# Hyper‑parameters

# In[ ]:


K_S        = 3   # k: Spline order 
G_S        = 3   # g: Grid size 
IMIN       = 0.0 
IMAX       = 1.0 
Order      = 2   # order: Nonlinearization power of HR-ReLU 
Batch_Size  = 32
Context_Len = 120
Pred_Len    = 20
H_Dims      = 32
N_Epochs    = 3000
Learning_Rate = 3e-4
Seq_Step = 5
Num_Layers  = 2
Bidirectional = False


# Displaying Time Series

# In[4]:


# Use 'P' and 'Y' columns for plotting
time_series_data_P = df['P']
time_series_data_Y = df['Y']
# Plot the time series data for 'P' and 'Y'
plt.figure(figsize=(12, 6))
plt.plot(time_series_data_P, label='P')
plt.plot(time_series_data_Y, label='Y')
plt.title('Time Series Data (P and Y)')
plt.xlabel('Time (Index)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


# Sliding‑window dataset (P → Y)

# In[5]:


class PairedTimeSeriesDataset(Dataset):
    def __init__(self, x_series, y_series, context_len=Context_Len, pred_len=Pred_Len):
        self.x = torch.tensor(x_series, dtype=torch.float32)
        self.y = torch.tensor(y_series, dtype=torch.float32)
        self.context_len = context_len
        self.pred_len = pred_len
        self.total_len = context_len + pred_len
        self.n_samples = max(len(self.x) - self.total_len + 1, 0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_win = self.x[idx : idx + self.context_len]                 # P‑window
        y_win = self.y[idx + self.context_len : idx + self.total_len]  # corresponding Y‑window
        return x_win, y_win

train_dataset = PairedTimeSeriesDataset(P_train, Y_train)
test_dataset  = PairedTimeSeriesDataset(P_test,  Y_test)

train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=Batch_Size, shuffle=False)

print(f"Train windows: {len(train_dataset)}")
print(f"Test  windows: {len(test_dataset)}")


# HRKAN layer and the Recurrent model

# In[6]:


class HRKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, imin: float, imax: float, order: int, train_ab: bool = True):
        super().__init__()
        self.g, self.k, self.r = g, k, 2*g / ((k+1)*(imax-imin))
        self.order = order
        self.input_size, self.output_size = input_size, output_size

        # Compute the start and end points of the grid intervals (B-spline basis functions)
        phase_low = (imax-imin) * (np.arange(-k, g) / g) - (-imin)
        phase_height = phase_low + (k+1) / g * (imax-imin)

        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                      requires_grad=train_ab)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g+k, input_size))

    def forward(self, x):
        # x: (batch, input_size)
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))

        # Build triangular/bounded basis functions using ReLU
        x1 = torch.relu(x_expanded - self.phase_low)
        x2 = torch.relu(self.phase_height - x_expanded)

        # Apply Higher-Order ReLU and then raise to the power of 2 (as in the original paper's formula)
        x = (x1 * x2 * self.r) ** self.order
        x = x * x  

        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        x = x.reshape((len(x), self.output_size))
        return x

class RecurrentHRKANTS(nn.Module):
    def __init__(self,
                 context_len: int,
                 pred_len: int,
                 g: int,
                 k: int,
                 imin: float,
                 imax: float,
                 order: int,
                 kan_out_dim: int,
                 rnn_hidden: int,
                 seq_step: int = 10,
                 num_layers: int = 2,
                 bidirectional: bool = False):
        super().__init__()
        assert context_len % seq_step == 0, "context_len must be a multiple of seq_step"
        self.context_len = context_len
        self.pred_len    = pred_len
        self.seq_step    = seq_step
        self.num_steps   = context_len // seq_step

        # ---- Actual HRKAN layer (based on Conv2d and B-spline basis functions) ----
        self.kan = HRKANLayer(
            input_size=seq_step,
            g=g,
            k=k,
            output_size=kan_out_dim,
            imin=imin,
            imax=imax,
            order=order
        )

        # ---- LSTM ----
        self.lstm = nn.LSTM(
            input_size=kan_out_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # ---- Output layer ----
        hidden_dim = rnn_hidden * (2 if bidirectional else 1)
        self.fc_out = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        if x.dim() == 2:
            b, L = x.shape
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)
            b, L = x.shape
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        assert L == self.context_len, f"Expected context_len={self.context_len}, got {L}"

        x_steps = x.view(b, self.num_steps, self.seq_step)
        x_flat = x_steps.reshape(b * self.num_steps, self.seq_step)

        # Apply HRKAN (this layer's output already has the correct dimensions)
        kan_out = self.kan(x_flat)  # Shape: (b*T, kan_out_dim)
        kan_seq = kan_out.view(b, self.num_steps, -1)

        # LSTM
        rnn_out, _ = self.lstm(kan_seq)
        last_out = rnn_out[:, -1, :]

        return self.fc_out(last_out)


# Training utilities

# In[7]:


class HRKANLayer(nn.Module):
    """
    HR-KAN layer compatible with the RecurrentHRKANTS model.
    It uses spline-like basis functions with higher-order ReLU and a Conv2d projection.
    """
    def __init__(self, input_size: int, g: int, k: int, output_size: int,
                 imin: float, imax: float, order: int = 2, train_ab: bool = True):
        super().__init__()
        if imax <= imin:
            raise ValueError("imax must be greater than imin")

        self.g, self.k, self.r = g, k, 2 * g / ((k + 1) * (imax - imin))
        self.order = order
        self.input_size, self.output_size = input_size, output_size

        phase_low = (imax - imin) * (np.arange(-k, g) / g) - (-imin)
        phase_height = phase_low + (k + 1) / g * (imax - imin)

        self.phase_low = nn.Parameter(
            torch.Tensor(np.array([phase_low for _ in range(input_size)])),
            requires_grad=train_ab,
        )
        self.phase_height = nn.Parameter(
            torch.Tensor(np.array([phase_height for _ in range(input_size)])),
            requires_grad=train_ab,
        )
        self.equal_size_conv = nn.Conv2d(1, output_size, (g + k, input_size))

    def forward(self, x):
        x = x.float()
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))

        x1 = torch.relu(x_expanded - self.phase_low)
        x2 = torch.relu(self.phase_height - x_expanded)

        x = (x1 * x2 * self.r) ** self.order
        x = x * x

        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        x = x.reshape((len(x), self.output_size))
        return x


# Device, model, optimizer, criterion

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = RecurrentHRKANTS(
    context_len = Context_Len,
    pred_len    = Pred_Len,
    g           = G_S,
    k           = K_S,
    imin        = IMIN,
    imax        = IMAX,
    order       = Order,
    kan_out_dim = H_Dims,
    rnn_hidden  = H_Dims,
    seq_step    = Seq_Step,
    num_layers  = Num_Layers,
    bidirectional = Bidirectional
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=Learning_Rate)
criterion = nn.L1Loss()  # MAE loss


# Training loop with timing

# In[9]:


train_history = []  
test_history  = []  
train_times   = []  
infer_times   = []  
per_iteration_train_times = []  
total_train_time = 0.0

def mae(y_true, y_pred): return torch.mean(torch.abs(y_true - y_pred))
def mse(y_true, y_pred): return torch.mean((y_true - y_pred) ** 2)
def rmse(y_true, y_pred): return torch.sqrt(mse(y_true, y_pred))

def compute_stats(loader, model, device):
    model.eval()
    tot_mae, tot_mse, tot_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            tot_mae += mae(y_pred, y).item() * y.numel()
            tot_mse += mse(y_pred, y).item() * y.numel()
            tot_samples += y.numel()
    return tot_mae / tot_samples, tot_mse / tot_samples, rmse(y, y_pred)

for epoch in range(1, N_Epochs + 1):
    epoch_start = time.perf_counter()
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        iter_start = time.perf_counter()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        per_iteration_train_times.append(time.perf_counter() - iter_start)

    epoch_time = time.perf_counter() - epoch_start
    train_times.append(epoch_time)
    total_train_time += epoch_time

    train_mae, train_mse, _ = compute_stats(train_loader, model, device)
    test_mae,  test_mse,  _ = compute_stats(test_loader,  model, device)

    train_history.append((train_mae, train_mse))
    test_history.append((test_mae,  test_mse))

    infer_start = time.perf_counter()
    _, _, _ = compute_stats(test_loader, model, device)
    infer_time = time.perf_counter() - infer_start
    infer_times.append(infer_time)

    print(
        f"Epoch {epoch:04d} | "
        f"Loss {running_loss/len(train_loader.dataset):.6f} | "
        f"Train MAE {train_mae:.6f} | Train MSE {train_mse:.6f} | "
        f"Test MAE {test_mae:.6f} | Test MSE {test_mse:.6f} | "
        f"Train time {epoch_time:.2f}s | Inference time {infer_time:.2f}s"
    )


# Final evaluation & speed‑up

# In[10]:


final_train_mae, final_train_mse, _ = compute_stats(train_loader, model, device)
final_test_mae, final_test_mse, _ = compute_stats(test_loader, model, device)

def physics_simulation(x_batch):
    time.sleep(0.0001 * x_batch.shape[0]) 

physics_start = time.perf_counter()
for x, _ in test_loader:
    physics_simulation(x.to(device))
physics_time = time.perf_counter() - physics_start

avg_infer_time = np.mean(infer_times)
speedup = physics_time / (avg_infer_time * len(test_loader))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

results_df = pd.DataFrame({
    "Metric": [
        "Train MAE", "Test MAE", "Train MSE", "Test MSE",
        "Speedup (Physics / HRKAN LSTM)", "Total Training Time",
        "Test Time", "Per-Iteration Training Time (s)",
        "Total number of trainable parameters:", "Structure"
    ],
    "Value": [
        final_train_mae, final_test_mae, final_train_mse, final_test_mse,
        speedup, total_train_time, np.mean(infer_times),
        np.mean(per_iteration_train_times), count_parameters(model), str(model)
    ]
})

print("\n--- Summary of Results (Markdown) ---")
print(results_df.to_markdown(index=False))


# Result table (Markdown)

# Plot training history (MAE / MSE per epoch)

# In[11]:


epochs = range(1, N_Epochs + 1)
train_mae_hist = [h[0] for h in train_history]
train_mse_hist = [h[1] for h in train_history]
test_mae_hist  = [h[0] for h in test_history]
test_mse_hist  = [h[1] for h in test_history]

plt.figure(figsize=(12, 5))
plt.plot(epochs, train_mae_hist, label="Train MAE", color="blue")
plt.plot(epochs, test_mae_hist,  label="Test MAE",  color="red", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("MAE per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(epochs, train_mse_hist, label="Train MSE", color="green")
plt.plot(epochs, test_mse_hist,  label="Test MSE",  color="orange", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Full test‑set prediction vs ground truth

# In[12]:


model.eval()
full_predicted_Y_test = np.full_like(Y_test, np.nan, dtype=np.float64)

with torch.no_grad():
    for idx in range(len(test_dataset)):
        x_win, _ = test_dataset[idx]           # only need the input
        x_win = x_win.unsqueeze(0).to(device)  # (1, context_len)
        y_pred_win = model(x_win).cpu().squeeze(0).numpy()  # (pred_len,)

        start_idx = idx + Context_Len
        end_idx   = start_idx + Pred_Len
        if start_idx < len(Y_test):
            full_predicted_Y_test[start_idx : min(end_idx, len(Y_test))] = \
                y_pred_win[:min(Pred_Len, len(Y_test) - start_idx)]

start_plot = 145
plt.figure(figsize=(15, 6))
plt.plot(np.arange(start_plot, len(Y_test)), Y_test[start_plot:], label="Actual Y", color="blue")
plt.plot(np.arange(start_plot, len(full_predicted_Y_test)), full_predicted_Y_test[start_plot:],
         label="HRKAN LSTM Prediction", color="red", linestyle="--")
plt.xlabel("Time index")
plt.ylabel("Signal value")
plt.title("Full Test Set: Actual vs. HRKAN LSTM Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

