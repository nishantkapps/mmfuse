import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import joblib
 
df = pd.read_csv("synthetic_data.csv")

df = df[df["command"] != "idle"].copy()
df = df.sort_values(["id", "t"])
#print(df)

# Command mapping
cmds = sorted(df["command"].unique())
cmd_to_id = {c: i for i, c in enumerate(cmds)}
df["cmd_id"] = df["command"].map(cmd_to_id)


train_losses = []
val_losses = []
train_accs = []
val_accs = []

class TrajectoryDataset(Dataset):

    def __init__(self, df, history_length=10, future_steps=20):

        self.X = []
        self.Y = []
        self.C = []

        for trajectory_id, group in df.groupby("id"):

            group = group.sort_values("t")

            states = group[
                ["X", "Y", "dx", "dy"]
            ].values.astype("float32")

            commands = group["cmd_id"].values.astype("int64")

            for i in range(
                history_length,
                len(group) - future_steps + 1
            ):

                # Past 10 states
                history = states[
                    i-history_length:i
                ]

                # Next 20 movements
                target = states[
                    i:i+future_steps,
                    2:4
                ]

                # Command
                command = commands[i]

                self.X.append(history)
                self.Y.append(target)
                self.C.append(command)

        self.X = torch.tensor(np.array(self.X))
        self.Y = torch.tensor(np.array(self.Y))
        self.C = torch.tensor(np.array(self.C))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.C[i], self.X[i], self.Y[i]


ids = df["id"].unique()

train_ids, temp_ids = train_test_split(
    ids,
    test_size=0.30,
    random_state=42
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.50,
    random_state=42
)

train_df = df[df["id"].isin(train_ids)].copy()
val_df   = df[df["id"].isin(val_ids)].copy()
test_df  = df[df["id"].isin(test_ids)].copy()

print(
    f"Trajectories: "
    f"train={len(train_ids)}, "
    f"val={len(val_ids)}, "
    f"test={len(test_ids)}"
)
pos_scaler = StandardScaler()
vel_scaler = StandardScaler()

pos_scaler.fit(train_df[["X", "Y"]])
vel_scaler.fit(train_df[["dx", "dy"]])

for d in [train_df, val_df]:
    d[["X", "Y"]] = pos_scaler.transform(d[["X", "Y"]])
    d[["dx", "dy"]] = vel_scaler.transform(d[["dx", "dy"]])
def trajectory_accuracy(pred, target, threshold=0.1):

    dist = torch.norm(pred - target, dim=2)

    # percentage of predicted timesteps within threshold
    return (dist < threshold).float().mean().item()

class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        num_commands,
        input_dim=4,
        d_model=128,
        nhead=4,
        num_layers=3,
        future_steps=20
    ):
        super().__init__()

        self.future_steps = future_steps

        # State -> transformer embedding
        self.input_proj = nn.Linear(input_dim, d_model)

        # Command embedding
        self.cmd_embed = nn.Embedding(num_commands, d_model)

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 100, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, future_steps * 2)
        )

    def forward(self, cmd, history):
        x = self.input_proj(history)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len]
        cmd_emb = self.cmd_embed(cmd)
        x = x + cmd_emb.unsqueeze(1)
        x = self.transformer(x)
        x = x[:, -1]
        out = self.head(x)
        return out.view(
            -1,
            self.future_steps,
            2
        )
history_length = 10
future_steps = 20

train_dataset = TrajectoryDataset(
    train_df,
    history_length,
    future_steps
)

val_dataset = TrajectoryDataset(
    val_df,
    history_length,
    future_steps
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_dataset = TrajectoryDataset(
    test_df,
    history_length,
    future_steps
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrajectoryTransformer(
    num_commands=len(cmd_to_id),
    future_steps=future_steps
).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)
loss_fn = nn.HuberLoss()

def train():

    max_epochs = 100
    patience = 10

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):

        # -------------------------
        # TRAIN
        # -------------------------
        model.train()

        train_loss = 0.0

        for cmd, history, target in train_loader:

            cmd = cmd.to(device)
            history = history.to(device)
            target = target.to(device)

            pred = model(cmd, history)

            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            train_loss += loss.item() * len(cmd)

        train_loss /= len(train_loader.dataset)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for cmd, history, target in val_loader:

                cmd = cmd.to(device)
                history = history.to(device)
                target = target.to(device)

                pred = model(cmd, history)

                loss = loss_fn(pred, target)

                val_loss += loss.item() * len(cmd)

        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f}"
        )

        # -------------------------
        # SAVE BEST MODEL
        # -------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                "best_trajectory_transformer.pt"
            )

        else:
            epochs_without_improvement += 1

        # -------------------------
        # EARLY STOPPING
        # -------------------------
        if epochs_without_improvement >= patience:

            print(
                f"\nEarly stopping at epoch {epoch+1}"
            )

            break

    # Load best validation model
    model.load_state_dict(
        torch.load(
            "best_trajectory_transformer.pt",
            map_location=device
        )
    )

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.4f}"
    )

def evaluate_trajectory(loader):

    model.eval()

    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    with torch.no_grad():

        for cmd, history, target in loader:

            cmd = cmd.to(device)
            history = history.to(device)
            target = target.to(device)

            pred = model(cmd, history)

            # Convert velocity predictions to positions
            # Both are currently in scaled velocity units,
            # so first convert velocity differences back to
            # physical units.

            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            pred_vel = vel_scaler.inverse_transform(
                pred_np.reshape(-1, 2)
            ).reshape(pred_np.shape)

            target_vel = vel_scaler.inverse_transform(
                target_np.reshape(-1, 2)
            ).reshape(target_np.shape)

            # Starting position is the final history position
            history_np = history.cpu().numpy()

            start_pos_scaled = history_np[:, -1, :2]

            start_pos = pos_scaler.inverse_transform(
                start_pos_scaled
            )

            pred_pos = (
                start_pos[:, None, :]
                + np.cumsum(pred_vel, axis=1)
            )

            target_pos = (
                start_pos[:, None, :]
                + np.cumsum(target_vel, axis=1)
            )

            distances = np.linalg.norm(
                pred_pos - target_pos,
                axis=2
            )

            ade = distances.mean(axis=1)
            fde = distances[:, -1]

            total_ade += ade.sum()
            total_fde += fde.sum()
            total_samples += len(cmd)

    return (
        total_ade / total_samples,
        total_fde / total_samples
    )
def analysis():
    plt.figure()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")


    plt.xlabel("Epoch")
    plt.title("Loss and Accuracy")
    plt.legend()
    plt.show()
    for cmd in df["command"].unique():
        subset = df[df["command"] == cmd]
        plt.scatter(subset["dx"], subset["dy"], label=cmd, alpha=0.3)

    plt.legend()
    plt.title("dx/dy distribution by command")
    plt.show()

    for cmd in ["left", "right"]:
        subset = df[df["command"] == cmd]

        plt.figure()
        plt.plot(subset["X"], subset["Y"], ".-")
        plt.title(cmd)
        plt.axis("equal")
        plt.show()

def predict_trajectory(command, history):

    model.eval()

    cmd_id = torch.tensor(
        [cmd_to_id[command]],
        dtype=torch.long,
        device=device
    )

    history = np.asarray(history, dtype=np.float32)

    # history shape: (10, 4)
    history_pos = pos_scaler.transform(history[:, :2])
    history_vel = vel_scaler.transform(history[:, 2:4])

    history_scaled = np.concatenate(
        [history_pos, history_vel],
        axis=1
    )

    history_tensor = torch.tensor(
        history_scaled,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        pred_deltas_scaled = model(
            cmd_id,
            history_tensor
        )[0].cpu().numpy()

    # Convert predicted dx/dy back to original units
    pred_deltas = vel_scaler.inverse_transform(
        pred_deltas_scaled
    )

    return pred_deltas

if __name__ == "__main__":
    train()
    test_ade, test_fde = evaluate_trajectory(test_loader)

    print("\n── Test Results ──────────────────")
    print(f"ADE: {test_ade:.4f}")
    print(f"FDE: {test_fde:.4f}")


    torch.save({
    "model": model.state_dict(),
    "cmd_to_id": cmd_to_id,
    "history_length": history_length,
    "future_steps": future_steps
    }, "trajectory_transformer.pt")

    joblib.dump(
    {
        "pos_scaler": pos_scaler,
        "vel_scaler": vel_scaler
    },
    "scalers.pkl"
    )

