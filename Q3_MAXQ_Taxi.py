# Q3: MAXQ Value Decomposition
# Taxi Navigation Environment
taxi_position = (0, 0)
passenger = (2, 2)
destination = (4, 4)
total_value = 0
def move_taxi(start, goal):
    """Move taxi toward the goal and calculate value."""
    x, y = start
    gx, gy = goal
    steps = 0
    value = 0
    while (x, y) != goal:
        if x < gx:
            x += 1
        elif x > gx:
            x -= 1
        elif y < gy:
            y += 1
        elif y > gy:
            y -= 1
        steps += 1
        value -= 1
    return (x, y), steps, value
print("MAXQ TAXI NAVIGATION")
print("---------------------")
# Subtask 1: Navigate to passenger
taxi_position, steps, value = move_taxi(
    taxi_position, passenger
)
print("\nSubtask 1: Navigate to Passenger")
print("Taxi Position:", taxi_position)
print("Steps:", steps)
print("Value:", value)
total_value += value
# Subtask 2: Pick up passenger
pickup_reward = 10
print("\nSubtask 2: Pick Up Passenger")
print("Passenger picked up.")
print("Value:", pickup_reward)
total_value += pickup_reward
# Subtask 3: Navigate to destination
taxi_position, steps, value = move_taxi(
    taxi_position, destination
)
print("\nSubtask 3: Navigate to Destination")
print("Taxi Position:", taxi_position)
print("Steps:", steps)
print("Value:", value)
total_value += value
# Subtask 4: Drop passenger
drop_reward = 20
print("\nSubtask 4: Drop Passenger")
print("Passenger dropped successfully.")
print("Value:", drop_reward)
total_value += drop_reward
print("\nRESULT")
print("Taxi reached the destination.")
print("MAXQ Total Value:", total_value)