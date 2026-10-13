import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# PPO Policy
ppo = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

# A2C Actor
a2c_actor = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

# A2C Critic
a2c_critic = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])


def simulate(policy, name):

    position = 0.0
    total_reward = 0
    total_deviation = 0
    safe_steps = 0

    print("\n" + name)
    print("-------------------------")

    for step in range(15):

        state = np.array(
            [[position / 5.0]],
            dtype=np.float32
        )

        probabilities = policy(state).numpy()[0]

        action = np.argmax(probabilities)

        # Actions:
        # 0 = LEFT
        # 1 = STRAIGHT
        # 2 = RIGHT

        if action == 0:
            position -= 0.3
            action_name = "LEFT"

        elif action == 1:
            action_name = "STRAIGHT"

        else:
            position += 0.3
            action_name = "RIGHT"

        deviation = abs(position)

        # Reward based on lane deviation
        reward = 10 - deviation * 5

        if deviation <= 1:
            safe_steps += 1
        else:
            reward -= 10

        total_reward += reward
        total_deviation += deviation

        print(
            "Step:", step + 1,
            "| Action:", action_name,
            "| Position:", round(position, 2),
            "| Deviation:", round(deviation, 2),
            "| Reward:", round(reward, 2)
        )

    average_deviation = total_deviation / 15

    return average_deviation, safe_steps, total_reward


print("PPO AUTONOMOUS LANE-KEEPING")
print("---------------------------")


# PPO Training
print("\nTraining PPO...")

for episode in range(10):

    state = np.array(
        [[np.random.uniform(-1, 1)]],
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


# A2C Training
print("\nTraining A2C...")

for episode in range(10):

    state = np.array(
        [[np.random.uniform(-1, 1)]],
        dtype=np.float32
    )

    with tf.GradientTape() as tape:

        probability = a2c_actor(state)
        value = a2c_critic(state)

        actor_loss = -tf.math.log(
            tf.reduce_max(probability) + 1e-7
        )

        critic_loss = tf.square(value)

        loss = actor_loss + critic_loss

    variables = (
        a2c_actor.trainable_variables
        + a2c_critic.trainable_variables
    )

    gradients = tape.gradient(loss, variables)

    for variable, gradient in zip(
        variables,
        gradients
    ):
        variable.assign_sub(
            0.001 * gradient
        )

print("A2C training completed.")


# Evaluation
ppo_deviation, ppo_safe, ppo_reward = simulate(
    ppo,
    "PPO RESULTS"
)

a2c_deviation, a2c_safe, a2c_reward = simulate(
    a2c_actor,
    "A2C RESULTS"
)


print("\nFINAL COMPARISON")
print("-------------------------")

print("PPO Average Lane Deviation:",
      round(ppo_deviation, 2))

print("PPO Safe Steps:",
      ppo_safe)

print("PPO Total Reward:",
      round(ppo_reward, 2))

print("\nA2C Average Lane Deviation:",
      round(a2c_deviation, 2))

print("A2C Safe Steps:",
      a2c_safe)

print("A2C Total Reward:",
      round(a2c_reward, 2))

if ppo_reward >= a2c_reward:
    print("\nPPO performed better in this simulation.")
else:
    print("\nA2C performed better in this simulation.")

print("Lane-keeping simulation completed.")