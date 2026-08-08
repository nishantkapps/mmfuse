import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("labeled_trajectory.csv")

df = df[df["command"] != "idle"].copy()
df = df.sort_values(["segment_id", "t"])
threshold = 2.0  # tune this

df = df[(abs(df["dx"]) > threshold) | (abs(df["dy"]) > threshold)]

# Command mapping
cmds = sorted(df["command"].unique())
cmd_to_id = {c: i for i, c in enumerate(cmds)}
df["cmd_id"] = df["command"].map(cmd_to_id)

df["dx_prev"] = df.groupby("segment_id")["dx"].shift(1).fillna(0)
df["dy_prev"] = df.groupby("segment_id")["dy"].shift(1).fillna(0)

X = df[["X", "Y", "dx_prev", "dy_prev"]].values.astype("float32")
Y = df[["dx", "dy"]].values.astype("float32")
C = df["cmd_id"].values.astype("int64")

scaler = StandardScaler()
X[:, :2] = scaler.fit_transform(X[:, :2])

X_tr, X_val, Y_tr, Y_val, C_tr, C_val = train_test_split(
    X, Y, C, test_size=0.15, random_state=42, stratify=C
)
df["dx_prev"] = df.groupby("segment_id")["dx"].shift(1).fillna(0)
df["dy_prev"] = df.groupby("segment_id")["dy"].shift(1).fillna(0)

X = df[["X", "Y", "dx_prev", "dy_prev"]].values.astype("float32")

train_losses = []
val_losses = []
train_accs = []
val_accs = []

def movement_accuracy(pred, target, threshold=0.1):
    # threshold in pixels (tune this)
    dist = torch.norm(pred - target, dim=1)
    return (dist < threshold).float().mean().item()

class Controller(nn.Module):
    def __init__(self, num_commands):
        super().__init__()
        self.embed = nn.Embedding(num_commands, 16)

        self.net = nn.Sequential(
            nn.Linear(16 + 4, 64),  # ← changed from 2 → 4
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, cmd, pos):
        emb = self.embed(cmd)
        x = torch.cat([emb, pos], dim=1)
        return self.net(x)

class ArmDataset(Dataset):
    def __init__(self, X, Y, C):
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
        self.C = torch.tensor(C)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.C[i], self.X[i], self.Y[i]

train_loader = DataLoader(ArmDataset(X_tr, Y_tr, C_tr), batch_size=64, shuffle=True)
val_loader   = DataLoader(ArmDataset(X_val, Y_val, C_val), batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Controller(num_commands=len(cmd_to_id)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.HuberLoss()

for epoch in range(1000):
    model.train()
    train_loss = 0
    train_acc = 0

    for cmd, pos, target in train_loader:
        cmd, pos, target = cmd.to(device), pos.to(device), target.to(device)

        pred = model(cmd, pos)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item() * len(cmd)
        train_acc += movement_accuracy(pred, target) * len(cmd)

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    # validation
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for cmd, pos, target in val_loader:
            cmd, pos, target = cmd.to(device), pos.to(device), target.to(device)

            pred = model(cmd, pos)
            val_loss += loss_fn(pred, target).item() * len(cmd)
            val_acc += movement_accuracy(pred, target) * len(cmd)

    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)

    # store
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")


plt.figure()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")


plt.xlabel("Epoch")
plt.title("Loss and Accuracy")
plt.legend()
plt.show()

def rollout(command, start_pos, steps=20):
    model.eval()

    cmd_id = torch.tensor([cmd_to_id[command]]).to(device)

    # initial position
    pos = scaler.transform([start_pos])[0]

    # initial velocity = 0
    vel = np.array([0.0, 0.0], dtype=np.float32)

    pos = torch.tensor(pos, dtype=torch.float32).unsqueeze(0).to(device)
    vel = torch.tensor(vel, dtype=torch.float32).unsqueeze(0).to(device)

    traj_scaled = []

    for _ in range(steps):
        with torch.no_grad():
            inp = torch.cat([pos, vel], dim=1)  # (1, 4)
            dxdy = model(cmd_id, inp)
            dxdy_scaled = dxdy / scaler.scale_[:2]


        # update position + velocity
        pos = pos + dxdy_scaled
        vel = dxdy_scaled

        traj_scaled.append(pos.cpu().numpy()[0])

    traj_scaled = np.array(traj_scaled)
    traj = scaler.inverse_transform(traj_scaled)

    return traj



torch.save({
    "model": model.state_dict(),
    "cmd_to_id": cmd_to_id
}, "controller.pt")

joblib.dump(scaler, "scaler.pkl")


for cmd in df["command"].unique():
    subset = df[df["command"] == cmd]
    plt.scatter(subset["dx"], subset["dy"], label=cmd, alpha=0.3)

plt.legend()
plt.title("dx/dy distribution by command")
plt.show()