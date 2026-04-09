import torch 
import torch.nn as nn
import numpy as np
import torch.optim as optim

class GripperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),   
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)    
        )

    def forward(self, x):
        return self.net(x)

def predict_next_position(model, pressure, command, x, y):
    model.eval()

    inp = torch.tensor([[pressure, command, x, y]], dtype=torch.float32)

    with torch.no_grad():
        dx, dy = model(inp).numpy()[0]

    new_x = x + dx
    new_y = y + dy

    return new_x, new_y, dx, dy

# LOAD
loaded_model = GripperModel()
loaded_model.load_state_dict(torch.load("gripper_model.pth"))
loaded_model.eval()

# USE
pressure = 0.6
command = 1
x, y = 0.0, 0.0

new_x, new_y, dx, dy = predict_next_position(
    loaded_model, pressure, command, x, y
)

print(new_x, new_y)