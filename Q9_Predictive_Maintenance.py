import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# REINFORCE Policy
reinforce = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

# PPO Policy
ppo = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])


def simulate(policy, name):

    health = 100
    total_cost = 0
    downtime = 0
    total_reward = 0

    print("\n" + name)
    print("-------------------------")

    for step in range(15):

        state = np.array(
            [[health / 100.0]],
            dtype=np.float32
        )

        probability = policy(state).numpy()[0]

        action = np.argmax(probability)

        # Actions
        if action == 0:
            action_name = "CONTINUE"
            cost = 0
            health -= 5
            down = 0

        elif action == 1:
            action_name = "PREVENTIVE MAINTENANCE"
            cost = 5
            health += 15
            down = 1

        else:
            action_name = "IMMEDIATE MAINTENANCE"
            cost = 10
            health += 25
            down = 2

        # Keep health within limits
        health = min(100, max(0, health))

        # Reward
        reward = health * 0.1 - cost - down * 2

        total_cost += cost
        downtime += down
        total_reward += reward

        print(
            "Step:", step + 1,
            "| Action:", action_name,
            "| Health:", health,
            "| Cost:", cost,
            "| Downtime:", down
        )

    return total_cost, downtime, total_reward


print("POLICY GRADIENT PREDICTIVE MAINTENANCE")
print("--------------------------------------")


# REINFORCE Training
print("\nTraining REINFORCE...")

for episode in range(10):

    state = np.array(
        [[np.random.uniform(0.3, 1.0)]],
        dtype=np.float32
    )

    with tf.GradientTape() as tape:

        probability = reinforce(state)

        loss = -tf.math.log(
            tf.reduce_max(probability) + 1e-7
        )

    gradients = tape.gradient(
        loss,
        reinforce.trainable_variables
    )

    for variable, gradient in zip(
        reinforce.trainable_variables,
        gradients
    ):
        variable.assign_sub(
            0.001 * gradient
        )

print("REINFORCE training completed.")


# PPO Training
print("\nTraining PPO...")

for episode in range(10):

    state = np.array(
        [[np.random.uniform(0.3, 1.0)]],
        dtype=np.float32
    )

    with tf.GradientTape() as tape:

        probability = ppo(state)

        loss = -tf.math.log(
            tf.reduce_max(probability) + 1e-7
        )

    gradients = tape.gradient(
        loss,
        ppo.trainable_variables
    )

    for variable, gradient in zip(
        ppo.trainable_variables,
        gradients
    ):
        variable.assign_sub(
            0.001 * gradient
        )

print("PPO training completed.")


# Evaluation
reinforce_cost, reinforce_down, reinforce_reward = simulate(
    reinforce,
    "REINFORCE RESULTS"
)

ppo_cost, ppo_down, ppo_reward = simulate(
    ppo,
    "PPO RESULTS"
)


print("\nFINAL COMPARISON")
print("-------------------------")

print("REINFORCE Maintenance Cost:",
      reinforce_cost)

print("REINFORCE Downtime:",
      reinforce_down)

print("REINFORCE Reward:",
      round(reinforce_reward, 2))

print("\nPPO Maintenance Cost:",
      ppo_cost)

print("PPO Downtime:",
      ppo_down)

print("PPO Reward:",
      round(ppo_reward, 2))


if ppo_reward >= reinforce_reward:
    print("\nPPO performed better in this simulation.")
else:
    print("\nREINFORCE performed better in this simulation.")

print("Predictive maintenance simulation completed.")