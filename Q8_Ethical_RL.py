# Q8: Ethical Reinforcement Learning
# Safety-Constrained Autonomous Decision Making

position = 0
goal = 10

# Dangerous positions
danger_zone = [4, 5, 6]

total_reward = 0
steps = 0

print("ETHICAL REINFORCEMENT LEARNING")
print("-----------------------------")

while position < goal and steps < 20:

    steps += 1

    next_position = position + 1

    print("\nCurrent Position:", position)

    # Safety constraint
    if next_position in danger_zone:

        print("Danger detected at position", next_position)
        print("Action: WAIT / AVOID")

        # Move around the danger zone
        position = 7
        reward = -1

    else:

        position = next_position
        reward = 5

        print("Action: MOVE_FORWARD")

    total_reward += reward

    print("New Position:", position)
    print("Reward:", reward)


print("\nRESULT")
print("-----------------------------")

if position >= goal:
    print("Goal reached safely.")
else:
    print("Goal not reached.")

print("Final Position:", position)
print("Total Reward:", total_reward)
print("Safety constraint was maintained.")