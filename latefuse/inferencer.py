import torch
import joblib
import numpy as np

from latefusemodel import Controller
import serial
import time



# load
checkpoint = torch.load("controller.pt", map_location="cpu")
scaler = joblib.load("scaler.pkl")

cmd_to_id = checkpoint["cmd_to_id"]

model = Controller(num_commands=len(cmd_to_id))
model.load_state_dict(checkpoint["model"])
model.eval()

def predict_step(command, pos, vel):
    """
    pos: [X, Y] (REAL pixel coords)
    vel: [dx, dy] (previous step)
    """

    cmd_id = torch.tensor([cmd_to_id[command]])

    # scale ONLY position
    pos_scaled = scaler.transform([pos])[0]

    inp = np.concatenate([pos_scaled, vel])  # (4,)

    inp = torch.tensor([inp], dtype=torch.float32)

    with torch.no_grad():
        dxdy = model(cmd_id, inp).numpy()[0]

    return dxdy

def rollout(command, start_pos, steps=20):
    pos = np.array(start_pos, dtype=np.float32)
    vel = np.array([0.0, 0.0], dtype=np.float32)

    traj = [pos.copy()]

    for _ in range(steps):
        dxdy = predict_step(command, pos, vel)

        pos = pos + dxdy
        vel = 0.8 * vel + 0.2 * dxdy

        traj.append(pos.copy())

    return np.array(traj)

traj = rollout("left", [0, 0], steps=20)

print(traj[:5])
# Change COM port accordingly (Windows: COM3, Linux/Mac: /dev/ttyUSB0)
ser = serial.Serial('COM3', 9600, timeout=1)

time.sleep(2)  # allow Arduino to reset

traj = rollout("left", [0, 0], steps=20)

for i in range(1, len(traj)):
    dx = traj[i][0] - traj[i-1][0]
    dy = traj[i][1] - traj[i-1][1]
    dx = np.clip(dx, -1.0, 1.0)
    dy = np.clip(dy, -1.0, 1.0)
    dx *= 0.05
    dy *= 0.05
    msg = f"{dx:.4f},{dy:.4f}\n"
    ser.write(msg.encode())

    print("Sent:", msg.strip())

    time.sleep(0.1)  # control speed (VERY IMPORTANT)

ser.close()
import matplotlib.pyplot as plt

traj = rollout("left", [0, 0], steps=20)

x = traj[:, 0]
y = traj[:, 1]

plt.figure()

# Main trajectory line
plt.plot(x, y, linewidth=2)

# Scatter points (fading over time)
plt.scatter(x, y, c=range(len(x)), cmap='viridis')

# Start point (green)
plt.scatter(x[0], y[0], s=100, marker='o', label='Start')

# End point (red)
plt.scatter(x[-1], y[-1], s=100, marker='X', label='End')

# Direction arrows (every few steps)
for i in range(0, len(x)-1, 3):
    plt.arrow(
        x[i], y[i],
        x[i+1] - x[i],
        y[i+1] - y[i],
        head_width=0.05,
        length_includes_head=True
    )

# Labels & title
plt.title("Predicted Trajectory (Command: LEFT)")
plt.xlabel("X Position")
plt.ylabel("Y Position")

# Equal scaling (VERY IMPORTANT)
plt.axis('equal')


plt.legend()
plt.grid(True)

plt.show()