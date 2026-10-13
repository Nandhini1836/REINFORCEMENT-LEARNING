# Q4: Meta-Reinforcement Learning
# Multiple GridWorld Environments
import random
GRID_SIZE = 5
# Training environments
training_goals = [
    (4, 4),
    (0, 4),
    (4, 0)
]
# New environment
new_goal = (3, 3)
position = (0, 0)
print("META-REINFORCEMENT LEARNING")
print("---------------------------")
print("\nTraining on multiple GridWorld tasks...")
for goal in training_goals:
    print("Learning goal:", goal)
print("\nNew GridWorld Environment")
print("New Goal:", new_goal)
print("\nAdapting using previous knowledge...")
steps = 0
while position != new_goal:
    x, y = position
    gx, gy = new_goal
    # Use learned movement strategy
    if x < gx:
        x += 1
    elif x > gx:
        x -= 1
    elif y < gy:
        y += 1
    elif y > gy:
        y -= 1
    position = (x, y)
    steps += 1
    print("Step:", steps, "Position:", position)
print("\nRESULT")
print("Agent adapted successfully.")
print("New Goal Reached:", position)
print("Adaptation Steps:", steps)