import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# A3C Actor
actor = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

# A3C Critic
critic = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Vanilla Policy Gradient
vpg = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])


def simulate(policy, name):

    total_response = 0
    total_cpu = 0
    total_reward = 0

    print("\n" + name)
    print("-------------------------")

    for step in range(10):

        # Simulated cloud workload
        tasks = np.random.randint(10, 50)
        cpu = np.random.randint(30, 90)

        state = np.array(
            [[tasks / 50, cpu / 100]],
            dtype=np.float32
        )

        probabilities = policy(state).numpy()[0]

        action = np.argmax(probabilities)

        # Resource allocation
        if action == 0:
            resource = "LOW"
            response_time = tasks * 1.2
            cpu_usage = min(cpu, 50)

        elif action == 1:
            resource = "MEDIUM"
            response_time = tasks * 0.8
            cpu_usage = min(cpu + 10, 80)

        else:
            resource = "HIGH"
            response_time = tasks * 0.5
            cpu_usage = min(cpu + 20, 100)

        reward = 100 / (response_time + 1) - cpu_usage * 0.05

        total_response += response_time
        total_cpu += cpu_usage
        total_reward += reward

        print(
            "Step:", step + 1,
            "| Tasks:", tasks,
            "| Resource:", resource,
            "| Response:", round(response_time, 2),
            "| CPU:", cpu_usage
        )

    avg_response = total_response / 10
    avg_cpu = total_cpu / 10

    return avg_response, avg_cpu, total_reward


print("A3C CLOUD RESOURCE ALLOCATION")
print("--------------------------------")

# Simple A3C training
print("\nTraining A3C...")

for episode in range(10):

    state = np.array(
        [[np.random.uniform(0.2, 1.0),
          np.random.uniform(0.2, 1.0)]],
        dtype=np.float32
    )

    with tf.GradientTape() as tape:

        action_probability = actor(state)
        value = critic(state)

        actor_loss = -tf.math.log(
            tf.reduce_max(action_probability) + 1e-7
        )

        critic_loss = tf.square(value)

        loss = actor_loss + critic_loss

    variables = (
        actor.trainable_variables
        + critic.trainable_variables
    )

    gradients = tape.gradient(loss, variables)

    for variable, gradient in zip(variables, gradients):

        variable.assign_sub(
            0.001 * gradient
        )

print("A3C training completed.")


# VPG training
print("\nTraining Vanilla Policy Gradient...")

for episode in range(10):

    state = np.array(
        [[np.random.uniform(0.2, 1.0),
          np.random.uniform(0.2, 1.0)]],
        dtype=np.float32
    )

    with tf.GradientTape() as tape:

        probability = vpg(state)

        loss = -tf.math.log(
            tf.reduce_max(probability) + 1e-7
        )

    gradients = tape.gradient(
        loss,
        vpg.trainable_variables
    )

    for variable, gradient in zip(
        vpg.trainable_variables,
        gradients
    ):

        variable.assign_sub(
            0.001 * gradient
        )

print("VPG training completed.")


# Evaluation
a3c_response, a3c_cpu, a3c_reward = simulate(
    actor,
    "A3C RESULTS"
)

vpg_response, vpg_cpu, vpg_reward = simulate(
    vpg,
    "VPG RESULTS"
)


print("\nFINAL COMPARISON")
print("-------------------------")

print("A3C Average Response Time:",
      round(a3c_response, 2))

print("A3C Average CPU Utilization:",
      round(a3c_cpu, 2))

print("A3C Total Reward:",
      round(a3c_reward, 2))

print("\nVPG Average Response Time:",
      round(vpg_response, 2))

print("VPG Average CPU Utilization:",
      round(vpg_cpu, 2))

print("VPG Total Reward:",
      round(vpg_reward, 2))

if a3c_reward >= vpg_reward:
    print("\nA3C performed better in this simulation.")
else:
    print("\nVPG performed better in this simulation.")