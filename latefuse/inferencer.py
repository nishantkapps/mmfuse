import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

from transformermodel import TrajectoryTransformer


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    "trajectory_transformer.pt",
    map_location="cpu"
)

scalers = joblib.load("scalers.pkl")

pos_scaler = scalers["pos_scaler"]
vel_scaler = scalers["vel_scaler"]

cmd_to_id = checkpoint["cmd_to_id"]

history_length = checkpoint["history_length"]
future_steps = checkpoint["future_steps"]


model = TrajectoryTransformer(
    num_commands=len(cmd_to_id),
    future_steps=future_steps
)

model.load_state_dict(checkpoint["model"])
model.eval()


# ============================================================
# CREATE FAKE HISTORY
# ============================================================

def generate_fake_history(command, history_length=10):

    X = 0.0
    Y = 0.0

    positions = []
    velocities = []

    for _ in range(history_length):

        if command == "right":
            dx = 0.05 + np.random.uniform(-0.01, 0.01)

        elif command == "left":
            dx = -0.05 + np.random.uniform(-0.01, 0.01)

        else:
            dx = 0.0

        dy = np.random.uniform(-0.01, 0.01)

        X += dx
        Y += dy

        positions.append([X, Y])
        velocities.append([dx, dy])

    history = np.concatenate(
        [
            np.array(positions, dtype=np.float32),
            np.array(velocities, dtype=np.float32)
        ],
        axis=1
    )

    return history


# ============================================================
# PREDICT TRAJECTORY
# ============================================================

def predict_trajectory(command, history):

    history = np.asarray(
        history,
        dtype=np.float32
    )

    # Scale positions
    history_pos = pos_scaler.transform(
        history[:, :2]
    )

    # Scale velocities
    history_vel = vel_scaler.transform(
        history[:, 2:4]
    )

    # Combine
    history_scaled = np.concatenate(
        [
            history_pos,
            history_vel
        ],
        axis=1
    )

    # (10, 4) -> (1, 10, 4)
    history_tensor = torch.tensor(
        history_scaled,
        dtype=torch.float32
    ).unsqueeze(0)

    # Command
    cmd_id = torch.tensor(
        [cmd_to_id[command]],
        dtype=torch.long
    )

    # Predict all 20 future movements
    with torch.no_grad():

        pred_deltas_scaled = model(
            cmd_id,
            history_tensor
        )[0].numpy()

    # Convert dx/dy back to original units
    pred_deltas = vel_scaler.inverse_transform(
        pred_deltas_scaled
    )

    # Starting position = last observed position
    start_position = history[-1, :2]

    # Convert predicted movements into positions
    pred_positions = (
        start_position
        + np.cumsum(pred_deltas, axis=0)
    )

    return pred_positions, pred_deltas


# ============================================================
# TEST
# ============================================================

command = "left"

history = generate_fake_history(
    command,
    history_length
)

pred_positions, pred_deltas = predict_trajectory(
    command,
    history
)


# ============================================================
# PRINT HISTORY
# ============================================================

print("\nObserved history:")

for i in range(history_length):

    print(
        f"History {i+1}: "
        f"X={history[i,0]:.3f}, "
        f"Y={history[i,1]:.3f}, "
        f"dx={history[i,2]:.3f}, "
        f"dy={history[i,3]:.3f}"
    )


# ============================================================
# PRINT PREDICTIONS
# ============================================================

print("\nPredicted future:")

for i in range(future_steps):

    print(
        f"Step {i+1}: "
        f"X={pred_positions[i,0]:.3f}, "
        f"Y={pred_positions[i,1]:.3f}, "
        f"dx={pred_deltas[i,0]:.3f}, "
        f"dy={pred_deltas[i,1]:.3f}"
    )


# ============================================================
# PLOT
# ============================================================

history_x = history[:, 0]
history_y = history[:, 1]

pred_x = pred_positions[:, 0]
pred_y = pred_positions[:, 1]

plt.figure(figsize=(8, 6))

# Observed history
plt.plot(
    history_x,
    history_y,
    "o-",
    label="Observed history"
)

# Predicted future
plt.plot(
    pred_x,
    pred_y,
    "o-",
    label="Predicted future"
)

# Current position
plt.scatter(
    history_x[-1],
    history_y[-1],
    s=120,
    marker="o",
    label="Current position"
)

# Final predicted position
plt.scatter(
    pred_x[-1],
    pred_y[-1],
    s=120,
    marker="X",
    label="Predicted end"
)

plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.title(
    f"Trajectory Prediction — Command: {command.upper()}"
)

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.savefig("trajectory_prediction.png", dpi=150, bbox_inches="tight")
plt.close()