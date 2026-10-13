import numpy as np
warehouse = np.zeros((6, 6))
start = (0, 0)
shelf = (2, 4)
packing = (5, 5)
position = start
total_reward = 0
path = [position]
# High-level policy
def high_level_policy(position):
    if position != shelf:
        return "GO_TO_SHELF"
    elif position != packing:
        return "GO_TO_PACKING"
    return "COMPLETED"
# Low-level policy
def low_level_policy(position, goal):
    x, y = position
    gx, gy = goal
    if x < gx:
        return (x + 1, y)
    elif x > gx:
        return (x - 1, y)
    elif y < gy:
        return (x, y + 1)
    elif y > gy:
        return (x, y - 1)
    return position
print("HRL WAREHOUSE NAVIGATION")
print("Start:", start)
print("Shelf:", shelf)
print("Packing:", packing)
step = 0
while position != packing:
    subtask = high_level_policy(position)
    if subtask == "GO_TO_SHELF":
        goal = shelf
    else:
        goal = packing
    print("\nSubtask:", subtask)
    while position != goal:
        position = low_level_policy(position, goal)
        path.append(position)
        step += 1
        reward = -1
        if position == goal:
            reward = 10
        total_reward += reward
        print("Step:", step,
              "Position:", position,
              "Reward:", reward)
print("\nRESULT")
print("Robot reached packing area.")
print("Total Steps:", step)
print("Total Reward:", total_reward)
print("Path:", path)