# Q10: RL-Based Smart Energy Management
# Microgrid Power Distribution

import random

battery = 20
total_reward = 0
total_grid_usage = 0

print("SMART ENERGY MANAGEMENT SYSTEM")
print("--------------------------------")

for hour in range(1, 11):

    demand = random.randint(10, 20)
    solar = random.randint(5, 15)

    print("\nHour:", hour)
    print("Energy Demand:", demand)
    print("Solar Energy:", solar)
    print("Battery Level:", battery)

    # Use solar power first
    solar_used = min(solar, demand)
    remaining = demand - solar_used

    # Use battery next
    battery_used = min(battery, remaining)
    battery -= battery_used
    remaining -= battery_used

    # Use grid for remaining demand
    grid_used = remaining
    total_grid_usage += grid_used

    # Charge battery with extra solar
    extra_solar = solar - solar_used
    battery += extra_solar

    # Calculate reward
    reward = solar_used * 2 + battery_used - grid_used * 2
    total_reward += reward

    print("Solar Used:", solar_used)
    print("Battery Used:", battery_used)
    print("Grid Used:", grid_used)
    print("New Battery Level:", battery)
    print("Reward:", reward)

print("\nRESULT")
print("--------------------------------")
print("Final Battery Level:", battery)
print("Total Grid Usage:", total_grid_usage)
print("Total Reward:", total_reward)

print("Energy distribution optimized successfully.")