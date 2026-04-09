import torch 
import torch.nn as nn
import numpy as np
import torch.optim as optim

class GripperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),   
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)    
        )

    def forward(self, x):
        return self.net(x)


def generate_dummy_data(n=1000):
    X = []
    Y = []

    for _ in range(n):
        pressure = np.random.uniform(0, 1)
        command = np.random.randint(0, 2)  
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)

        dx = 0.1 * pressure + 0.05 * command
        dy = 0.08 * pressure - 0.03 * command

        X.append([pressure, command])
        Y.append([dx, dy])

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X, Y = generate_dummy_data(2000)

X = torch.tensor(X)
Y = torch.tensor(Y)

model = GripperModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs = 50

for epoch in range(epochs):
    model.train()

    preds = model(X)
    loss = loss_fn(preds, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "gripper_model.pth")