import csv
import random
import os
import math

filename = "synthetic_data.csv"

num_trajectories = 100
steps_per_trajectory = 60

start_id = 6001

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

    # ---------------------------------------------------------
    # LEFTWARD TRAJECTORIES
    # ---------------------------------------------------------

    for _ in range(num_trajectories):

        # -----------------------------------------------------
        # RANDOM STARTING POSITION
        # -----------------------------------------------------

        X = random.uniform(0.8, 1.2)
        Y = random.uniform(-0.3, 0.3)

        # -----------------------------------------------------
        # RANDOM TRAJECTORY CHARACTERISTICS
        # -----------------------------------------------------

        # How far the end-effector travels left
        total_distance = random.uniform(1.8, 2.4)

        # Overall speed variation
        speed = random.uniform(0.85, 1.15)

        # Positive = upward curve
        # Negative = downward curve
        curvature = random.uniform(-0.08, 0.08)

        # How much the speed changes during the movement
        acceleration_strength = random.uniform(-0.35, 0.35)

        # Small random phase for trajectory variation
        phase = random.uniform(0, 2 * math.pi)

        # Smooth lateral oscillation
        lateral_amplitude = random.uniform(0.01, 0.05)

        # -----------------------------------------------------
        # CREATE A SPEED PROFILE
        # -----------------------------------------------------
        #
        # Different trajectories can:
        #
        #   accelerate
        #   decelerate
        #   remain relatively constant
        #
        # while still remaining smooth.

        raw_speeds = []

        for step in range(steps_per_trajectory):

            progress = step / (steps_per_trajectory - 1)

            # Base speed
            speed_profile = 1.0

            # Smooth acceleration/deceleration
            speed_profile += (
                acceleration_strength
                * (2 * progress - 1)
            )

            # Small smooth variation
            speed_profile += (
                0.08
                * math.sin(
                    2 * math.pi * progress + phase
                )
            )

            speed_profile *= speed

            raw_speeds.append(max(speed_profile, 0.1))

        # -----------------------------------------------------
        # NORMALIZE SPEEDS
        # -----------------------------------------------------
        #
        # This makes the total horizontal distance approximately
        # equal to total_distance regardless of the speed profile.

        total_speed = sum(raw_speeds)

        dx_values = [
            -total_distance * s / total_speed
            for s in raw_speeds
        ]

        # -----------------------------------------------------
        # GENERATE TRAJECTORY
        # -----------------------------------------------------

        t = 0.0

        for step in range(steps_per_trajectory):

            progress = step / (steps_per_trajectory - 1)

            dx = dx_values[step]

            # -------------------------------------------------
            # SMOOTH CURVATURE
            # -------------------------------------------------

            # Creates gradual movement upward/downward rather
            # than completely random dy values.

            curve_component = (
                curvature
                * math.sin(math.pi * progress)
            )

            # Small smooth lateral variation
            lateral_component = (
                lateral_amplitude
                * math.sin(
                    2 * math.pi * progress + phase
                )
            )

            # Small measurement/motion noise
            noise = random.gauss(0, 0.0015)

            dy = (
                curve_component
                + lateral_component
                + noise
            )

            # -------------------------------------------------
            # UPDATE POSITION
            # -------------------------------------------------

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


print(
    f"Generated {num_trajectories} "
    f"leftward trajectories."
)

print(f"Saved to {filename}")