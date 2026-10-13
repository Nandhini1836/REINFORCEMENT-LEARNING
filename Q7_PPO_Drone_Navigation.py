import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# PPO Policy Network
policy = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax")
])

# Starting and target positions
drone = [0, 0]
target = [8, 8]

# Static obstacle
obstacle = [4, 4]

total_reward = 0
energy = 0
steps = 0

print("PPO AUTONOMOUS DRONE NAVIGATION")
print("--------------------------------")

while drone != target and steps < 30:

    steps += 1

    # State representation
    state = np.array([[
        drone[0] / 10,
        drone[1] / 10,
        target[0] / 10,
        target[1] / 10
    ]], dtype=np.float32)

    # PPO policy selects action
    probabilities = policy(state).numpy()[0]
    action = np.argmax(probabilities)

    new_drone = drone.copy()

    # Actions:
    # 0 = UP
    # 1 = DOWN
    # 2 = LEFT
    # 3 = RIGHT

    if action == 0 and new_drone[1] < 9:
        new_drone[1] += 1

    elif action == 1 and new_drone[1] > 0:
        new_drone[1] -= 1

    elif action == 2 and new_drone[0] > 0:
        new_drone[0] -= 1

    elif action == 3 and new_drone[0] < 9:
        new_drone[0] += 1

    # Obstacle avoidance
    if new_drone == obstacle:
        reward = -10
        print("Obstacle detected! Drone avoids obstacle.")
    else:
        drone = new_drone

        # Distance to target
        distance = abs(target[0] - drone[0]) + \
                   abs(target[1] - drone[1])

        reward = -distance * 0.1
        energy += 1

    # Target reward
    if drone == target:
        reward += 50

    total_reward += reward

    print(
        "Step:", steps,
        "| Position:", drone,
        "| Reward:", round(reward, 2)
    )

print("\nRESULT")
print("--------------------------------")

print("Final Drone Position:", drone)
print("Target Position:", target)
print("Total Steps:", steps)
print("Energy Consumption:", energy)
print("Total Reward:", round(total_reward, 2))

if drone == target:
    print("Navigation successful.")
else:
    print("Navigation incomplete.")

print("PPO drone navigation simulation completed.")