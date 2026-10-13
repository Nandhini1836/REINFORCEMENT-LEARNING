# Q5: Cooperative Multi-Agent Reinforcement Learning
# Cooperative Robot Path Planning
robot1 = (0, 0)
robot2 = (0, 4)
goal1 = (4, 4)
goal2 = (4, 0)
reward1 = 0
reward2 = 0
print("COOPERATIVE MULTI-AGENT RL")
print("--------------------------")
step = 0
while robot1 != goal1 or robot2 != goal2:
    step += 1
    # Move Robot 1
    x, y = robot1
    gx, gy = goal1
    if x < gx:
        x += 1
    elif y < gy:
        y += 1
    new_robot1 = (x, y)
    # Move Robot 2
    x, y = robot2
    gx, gy = goal2
    if x < gx:
        x += 1
    elif y > gy:
        y -= 1
    new_robot2 = (x, y)
    # Collision check
    if new_robot1 == new_robot2:
        print("Collision detected! Robots adjust paths.")
        reward1 -= 5
        reward2 -= 5
    else:
        robot1 = new_robot1
        robot2 = new_robot2
    # Rewards
    if robot1 == goal1:
        reward1 += 10
    if robot2 == goal2:
        reward2 += 10
    print(
        "Step:", step,
        "| Robot 1:", robot1,
        "| Robot 2:", robot2
    )
    if robot1 == goal1 and robot2 == goal2:
        break
print("\nRESULT")
print("Robot 1 reached:", robot1)
print("Robot 2 reached:", robot2)
print("Robot 1 Reward:", reward1)
print("Robot 2 Reward:", reward2)
print("Both robots completed their tasks cooperatively.")