# Q6: Competitive Multi-Agent Reinforcement Learning
# Autonomous Drone Navigation
drone1 = (0, 0)
drone2 = (0, 4)
target = (4, 2)
reward1 = 0
reward2 = 0
print("COMPETITIVE MULTI-AGENT RL")
print("--------------------------")
step = 0
winner = None
while step < 20:
    step += 1
    # Move Drone 1 toward target
    x, y = drone1
    gx, gy = target
    if x < gx:
        drone1 = (x + 1, y)
    elif y < gy:
        drone1 = (x, y + 1)
    elif y > gy:
        drone1 = (x, y - 1)
    # Check Drone 1
    if drone1 == target:
        winner = "Drone 1"
        reward1 = 10
        reward2 = -10
        break
    # Move Drone 2 toward target
    x, y = drone2
    if x < gx:
        drone2 = (x + 1, y)
    elif y > gy:
        drone2 = (x, y - 1)
    elif y < gy:
        drone2 = (x, y + 1)
    # Check Drone 2
    if drone2 == target:
        winner = "Drone 2"
        reward2 = 10
        reward1 = -10
        break
    print(
        "Step:", step,
        "| Drone 1:", drone1,
        "| Drone 2:", drone2
    )
print("\nRESULT")
print("--------------------------")
print("Drone 1 Final Position:", drone1)
print("Drone 2 Final Position:", drone2)

if winner:
    print("Winner:", winner)
else:
    print("No drone reached the target.")

print("Drone 1 Reward:", reward1)
print("Drone 2 Reward:", reward2)