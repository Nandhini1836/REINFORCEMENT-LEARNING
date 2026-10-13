# Q9: RL-Based Adaptive Traffic Signal Control

import random

vehicles = 20
total_reward = 0

print("RL-BASED ADAPTIVE TRAFFIC SIGNAL")
print("--------------------------------")

for step in range(1, 11):

    # New vehicles arrive
    arrivals = random.randint(1, 5)
    vehicles += arrivals

    print("\nStep:", step)
    print("Vehicles waiting:", vehicles)

    # RL decision
    if vehicles > 15:
        action = "GREEN"
        passed = min(8, vehicles)
    else:
        action = "RED"
        passed = 0

    vehicles -= passed

    # Reward for reducing traffic
    reward = passed - vehicles * 0.1
    total_reward += reward

    print("Action:", action)
    print("Vehicles passed:", passed)
    print("Remaining vehicles:", vehicles)
    print("Reward:", round(reward, 2))

print("\nRESULT")
print("--------------------------------")
print("Remaining vehicles:", vehicles)
print("Total Reward:", round(total_reward, 2))

if vehicles < 10:
    print("Traffic was successfully reduced.")
else:
    print("Traffic remains high.")