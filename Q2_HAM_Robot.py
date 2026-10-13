# Q2: Hierarchical Abstract Machine (HAM)
# Sequential Robot Tasks
position = 0
total_reward = 0
tasks = [
    "MOVE_TO_SHELF",
    "PICK_ITEM",
    "MOVE_TO_PACKING",
    "DROP_ITEM"
]
print("HAM ROBOT TASK EXECUTION")
print("-------------------------")
for task in tasks:
    print("\nCurrent Task:", task)
    if task == "MOVE_TO_SHELF":
        position = 1
        print("Robot moved to shelf.")
        total_reward += 10
    elif task == "PICK_ITEM":
        print("Robot picked the item.")
        total_reward += 10
    elif task == "MOVE_TO_PACKING":
        position = 2
        print("Robot moved to packing area.")
        total_reward += 10
    elif task == "DROP_ITEM":
        print("Robot dropped the item.")
        total_reward += 10
print("\nRESULT")
print("All tasks completed successfully.")
print("Final Position:", position)
print("Total Reward:", total_reward)