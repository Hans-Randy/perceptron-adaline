import os
import io
import math
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 48
torch.manual_seed(SEED)
np.random.seed(SEED)

# Load Hepatitis dataset from UCI and clean it

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"

# Column names from UCI (class + 19 attributes)
cols = [
    "class", "age", "sex", "steroid", "antivirals", "fatigue", "malaise",
    "anorexia", "liver_big", "liver_firm", "spleen_palpable", "spiders",
    "ascites", "varices", "bilirubin", "alk_phosphate", "sgot", "albumin",
    "protime", "histology",
]

def load_hepatitis(url=UCI_URL):
    # The file uses commas, missing values are '?'
    raw = urllib.request.urlopen(url).read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), header=None, names=cols, na_values="?")
    
    # Splitting target and features
    y = df["class"].astype(int)  # 1=Die, 2=Live
    X = df.drop(columns=["class"]).copy()

    # Converting everything to float
    X = X.astype(float)

    # Imputing missing with column mean
    X = X.fillna(X.mean(numeric_only=True))

    # Some numeric columns might still be nan if whole column missing; guard:
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean(numeric_only=True))

    # Standardizing features 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # One-hot mapping
    # Die (1) -> [1, 0], Live (2) -> [0, 1]
    y_oh = np.zeros((len(y), 2), dtype=np.float32)
    y_oh[(y.values == 1), 0] = 1.0
    y_oh[(y.values == 2), 1] = 1.0

    return X_scaled.astype(np.float32), y_oh

X, y = load_hepatitis()

# Train/val/test split (60/20/20)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
val_ds   = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
test_ds  = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64)
test_loader  = DataLoader(test_ds, batch_size=64)

# Defining the MLP: 19-30-15-2 with sigmoid on ALL layers

class HepatitisMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(19, 30),
            nn.Sigmoid(),
            nn.Linear(30, 15),
            nn.Sigmoid(),
            nn.Linear(15, 2),
            # No activation on output layer for CrossEntropyLoss
        )

    def forward(self, x):
        return self.net(x)

model = HepatitisMLP()

# CrossEntropyLoss with class weights to handle imbalance
class_weights = torch.tensor([25.0/6.0, 1.0])  # Higher weight for minority class (Die)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Training

EPOCHS = 300
best_val = math.inf
patience = 40
wait = 0

def run_epoch(loader, train_flag=True):
    if train_flag:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    with torch.set_grad_enabled(train_flag):
        for xb, yb in loader:
            if train_flag:
                optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            if train_flag:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)

for epoch in range(1, EPOCHS + 1):
    tr_loss = run_epoch(train_loader, train_flag=True)
    va_loss = run_epoch(val_loader, train_flag=False)

    if va_loss < best_val - 1e-5:
        best_val = va_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f}")

    if wait >= patience:
        print(f"Early stopping at epoch {epoch} (no val improvement for {patience})")
        break

# Restore best model
model.load_state_dict(best_state)

# Evaluating

def predict_classes(loader):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)             # shape [B,2], raw logits
            pred_idx = logits.argmax(dim=1) # 0 => Die, 1 => Live
            true_idx = yb.argmax(dim=1)
            all_preds.append(pred_idx.cpu().numpy())
            all_true.append(true_idx.cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_preds)

y_true_test, y_pred_test = predict_classes(test_loader)
acc = accuracy_score(y_true_test, y_pred_test)
cm = confusion_matrix(y_true_test, y_pred_test, labels=[0,1])

print("\n=== Test Results ===")
print(f"Accuracy: {acc*100:.2f}%")
print("\nConfusion matrix (rows=true, cols=pred) [0=Die, 1=Live]:")
print(cm)
print("\nClassification report:")
print(classification_report(y_true_test, y_pred_test, target_names=["Die","Live"], zero_division=0))
