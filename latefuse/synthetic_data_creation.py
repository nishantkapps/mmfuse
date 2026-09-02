import csv
import random
import os

filename = "synthetic_data.csv"

# Number of trajectories of each type
num_trajectories = 100
steps_per_trajectory = 60

# Start ID after existing data
start_id = 6001

# Write header if the file doesn't exist yet
file_exists = os.path.exists(filename)

with open(filename, "a", newline="") as f:

    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "id",
            "t",
            "X",
            "Y",
            "dx",
            "dy",
            "command",
            "value"
        ])

    trajectory_id = start_id

    # RIGHT trajectories
    for _ in range(num_trajectories):

        X = 0.0
        Y = 0.0
        t = 0.0

        for step in range(steps_per_trajectory):

            # Generate movement
            dx = 0.05 + (random.random() - 0.5) * 0.02
            dy = (random.random() - 0.5) * 0.02

            X += dx
            Y += dy
            t += 0.34

            writer.writerow([
                trajectory_id,
                t,
                X,
                Y,
                dx,
                dy,
                "right",
                1.0
            ])

        trajectory_id += 1

    # LEFT trajectories
    for _ in range(num_trajectories):

        X = 0.0
        Y = 0.0
        t = 0.0

        for step in range(steps_per_trajectory):

            # Generate movement
            dx = -0.05 + (random.random() - 0.5) * 0.02
            dy = (random.random() - 0.5) * 0.02

            X += dx
            Y += dy
            t += 0.34

            writer.writerow([
                trajectory_id,
                t,
                X,
                Y,
                dx,
                dy,
                "left",
                1.0
            ])

        trajectory_id += 1

print(f"Generated {num_trajectories * 2} trajectories.")
print(f"Saved to {filename}")

