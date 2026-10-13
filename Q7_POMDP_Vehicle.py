# Q7: POMDP Autonomous Vehicle Navigation
position = 0
destination = 10
total_reward = 0

print("POMDP AUTONOMOUS VEHICLE NAVIGATION")
print("------------------------------------")

while position < destination:

    # Partial observation
    if position < 5:
        observation = "CLEAR"
    else:
        observation = "FOG"

    print("\nVehicle Position:", position)
    print("Observation:", observation)

    # Decision based on partial observation
    if observation == "CLEAR":
        action = "MOVE_FORWARD"
        position += 2
        reward = 5

    else:
        action = "MOVE_SLOWLY"
        position += 1
        reward = 3

    total_reward += reward

    print("Action:", action)
    print("Reward:", reward)

    # Safety condition
    if position > destination:
        position = destination

print("\nRESULT")
print("------------------------------------")
print("Vehicle reached destination.")
print("Final Position:", position)
print("Total Reward:", total_reward)