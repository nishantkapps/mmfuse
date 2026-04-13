import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

df = pd.read_csv("labeled_trajectory.csv")

# Create segment_id based on command changes
df["segment_id"] = (df["command"] != df["command"].shift()).cumsum()

# Remove idle segments if you want
df.loc[df["command"] == "idle", "segment_id"] = -1

# Optional: reassign clean IDs
valid = df["segment_id"] != -1
df.loc[valid, "segment_id"] = df.loc[valid, "segment_id"].factorize()[0]

# Save it back
df.to_csv("labeled_trajectory.csv", index=False)

# Remove idle rows
df = df[df["command"] != "idle"]

# Convert commands → numbers
cmd_to_id = {cmd: i for i, cmd in enumerate(df["command"].unique())}
df["cmd_id"] = df["command"].map(cmd_to_id)


scaler = StandardScaler()
df[["X", "Y"]] = scaler.fit_transform(df[["X", "Y"]])
grouped = df.groupby("segment_id")

X_data = []
Y_data = []

SEQ_LEN = 20  # fixed length

for seg_id, group in grouped:
    group = group.sort_values("t")

    cmd_id = group["cmd_id"].iloc[0]
    traj = group[["X", "Y"]].values

    # Skip very short segments
    if len(traj) < SEQ_LEN:
        continue

    # Resample / truncate
    traj = traj[:SEQ_LEN]

    X_data.append(cmd_id)
    Y_data.append(traj)

X_tensor = torch.tensor(X_data, dtype=torch.long)
Y_tensor = torch.tensor(Y_data, dtype=torch.float32)

class TrajectoryModel(nn.Module):
    def __init__(self, num_commands, embed_dim=16, hidden_dim=64):
        super().__init__()

        self.embed = nn.Embedding(num_commands, embed_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embed(x)              # (N, embed_dim)
        x = x.unsqueeze(1).repeat(1, SEQ_LEN, 1)

        out, _ = self.lstm(x)          # (N, SEQ_LEN, hidden)
        out = self.fc(out)             # (N, SEQ_LEN, 2)

        return out

model = TrajectoryModel(num_commands=len(cmd_to_id))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(50):
    pred = model(X_tensor)

    loss = loss_fn(pred, Y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

cmd = torch.tensor([cmd_to_id["left"]])

pred_traj = model(cmd).detach().numpy()