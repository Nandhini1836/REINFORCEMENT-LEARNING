import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)


class TrafficEnvironment:

    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.vehicles = np.random.randint(
            1, 10, size=4
        ).astype(np.float32)

        self.step_count = 0

        return self.get_state()

    def get_state(self):
        return self.vehicles / 20.0

    def step(self, action):

        self.step_count += 1

        arrivals = np.random.poisson(2, size=4)

        passed = 0

        if action == 0:

            ns_pass = (
                min(self.vehicles[0], 5) +
                min(self.vehicles[1], 5)
            )

            self.vehicles[0] -= min(
                self.vehicles[0], 5
            )

            self.vehicles[1] -= min(
                self.vehicles[1], 5
            )

            passed = int(ns_pass)

        else:

            ew_pass = (
                min(self.vehicles[2], 5) +
                min(self.vehicles[3], 5)
            )

            self.vehicles[2] -= min(
                self.vehicles[2], 5
            )

            self.vehicles[3] -= min(
                self.vehicles[3], 5
            )

            passed = int(ew_pass)

        self.vehicles += arrivals

        self.vehicles = np.clip(
            self.vehicles, 0, 20
        )

        waiting = np.sum(self.vehicles)

        reward = passed - 0.5 * waiting

        done = self.step_count >= self.max_steps

        return (
            self.get_state(),
            reward,
            done,
            passed,
            waiting
        )


class PolicyNetwork(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.d1 = tf.keras.layers.Dense(
            64, activation="relu"
        )

        self.d2 = tf.keras.layers.Dense(
            64, activation="relu"
        )

        self.output_layer = tf.keras.layers.Dense(
            2, activation="softmax"
        )

    def call(self, state):

        x = self.d1(state)
        x = self.d2(x)

        return self.output_layer(x)


def train_reinforce(episodes=300):

    model = PolicyNetwork()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )

    rewards_history = []
    waiting_history = []
    throughput_history = []

    for episode in range(episodes):

        env = TrafficEnvironment()

        state = env.reset()

        states = []
        actions = []
        rewards = []

        total_reward = 0
        total_waiting = 0
        total_throughput = 0

        done = False

        while not done:

            state_tensor = tf.convert_to_tensor(
                state.reshape(1, -1),
                dtype=tf.float32
            )

            probabilities = model(
                state_tensor
            )

            action = np.random.choice(
                2,
                p=probabilities.numpy()[0]
            )

            next_state, reward, done, passed, waiting = \
                env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            total_reward += reward
            total_waiting += waiting
            total_throughput += passed

            state = next_state

        returns = []

        G = 0

        for reward in reversed(rewards):

            G = reward + 0.99 * G

            returns.insert(0, G)

        returns = np.array(returns)

        returns = (
            returns - np.mean(returns)
        ) / (np.std(returns) + 1e-8)

        states_tensor = tf.convert_to_tensor(
            np.array(states),
            dtype=tf.float32
        )

        actions_tensor = tf.convert_to_tensor(
            actions,
            dtype=tf.int32
        )

        returns_tensor = tf.convert_to_tensor(
            returns,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:

            probabilities = model(
                states_tensor
            )

            selected_probabilities = tf.gather(
                probabilities,
                actions_tensor,
                axis=1,
                batch_dims=1
            )

            log_probabilities = tf.math.log(
                selected_probabilities + 1e-8
            )

            loss = -tf.reduce_mean(
                log_probabilities *
                returns_tensor
            )

        gradients = tape.gradient(
            loss,
            model.trainable_variables
        )

        optimizer.apply_gradients(
            zip(
                gradients,
                model.trainable_variables
            )
        )

        rewards_history.append(
            total_reward
        )

        waiting_history.append(
            total_waiting / env.max_steps
        )

        throughput_history.append(
            total_throughput
        )

        if (episode + 1) % 50 == 0:

            print(
                f"REINFORCE Episode {episode + 1} "
                f"Reward = {total_reward:.2f}"
            )

    return (
        model,
        rewards_history,
        waiting_history,
        throughput_history
    )


class A2CNetwork(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.d1 = tf.keras.layers.Dense(
            64, activation="relu"
        )

        self.d2 = tf.keras.layers.Dense(
            64, activation="relu"
        )

        self.actor = tf.keras.layers.Dense(
            2,
            activation="softmax"
        )

        self.critic = tf.keras.layers.Dense(
            1
        )

    def call(self, state):

        x = self.d1(state)
        x = self.d2(x)

        action_probabilities = self.actor(x)

        value = self.critic(x)

        return action_probabilities, value


def train_a2c(episodes=300):

    model = A2CNetwork()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )

    rewards_history = []
    waiting_history = []
    throughput_history = []

    for episode in range(episodes):

        env = TrafficEnvironment()

        state = env.reset()

        states = []
        actions = []
        rewards = []

        total_reward = 0
        total_waiting = 0
        total_throughput = 0

        done = False

        while not done:

            state_tensor = tf.convert_to_tensor(
                state.reshape(1, -1),
                dtype=tf.float32
            )

            probabilities, value = model(
                state_tensor
            )

            action = np.random.choice(
                2,
                p=probabilities.numpy()[0]
            )

            next_state, reward, done, passed, waiting = \
                env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            total_reward += reward
            total_waiting += waiting
            total_throughput += passed

            state = next_state

        returns = []

        G = 0

        for reward in reversed(rewards):

            G = reward + 0.99 * G

            returns.insert(0, G)

        returns = np.array(returns)

        states_tensor = tf.convert_to_tensor(
            np.array(states),
            dtype=tf.float32
        )

        actions_tensor = tf.convert_to_tensor(
            actions,
            dtype=tf.int32
        )

        returns_tensor = tf.convert_to_tensor(
            returns,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:

            probabilities, values = model(
                states_tensor
            )

            values = tf.squeeze(values)

            selected_probabilities = tf.gather(
                probabilities,
                actions_tensor,
                axis=1,
                batch_dims=1
            )

            log_probabilities = tf.math.log(
                selected_probabilities + 1e-8
            )

            advantages = (
                returns_tensor - values
            )

            actor_loss = -tf.reduce_mean(
                log_probabilities *
                tf.stop_gradient(advantages)
            )

            critic_loss = tf.reduce_mean(
                tf.square(advantages)
            )

            loss = (
                actor_loss +
                0.5 * critic_loss
            )

        gradients = tape.gradient(
            loss,
            model.trainable_variables
        )

        optimizer.apply_gradients(
            zip(
                gradients,
                model.trainable_variables
            )
        )

        rewards_history.append(
            total_reward
        )

        waiting_history.append(
            total_waiting / env.max_steps
        )

        throughput_history.append(
            total_throughput
        )

        if (episode + 1) % 50 == 0:

            print(
                f"A2C Episode {episode + 1} "
                f"Reward = {total_reward:.2f}"
            )

    return (
        model,
        rewards_history,
        waiting_history,
        throughput_history
    )


print("Training REINFORCE...")

reinforce_model, reinforce_rewards, \
reinforce_waiting, reinforce_throughput = \
    train_reinforce(300)


print("\nTraining A2C...")

a2c_model, a2c_rewards, \
a2c_waiting, a2c_throughput = \
    train_a2c(300)


print("\n========== RESULTS ==========")

print(
    "REINFORCE Average Waiting Time:",
    np.mean(reinforce_waiting[-50:])
)

print(
    "A2C Average Waiting Time:",
    np.mean(a2c_waiting[-50:])
)

print(
    "REINFORCE Average Throughput:",
    np.mean(reinforce_throughput[-50:])
)

print(
    "A2C Average Throughput:",
    np.mean(a2c_throughput[-50:])
)


plt.figure(figsize=(10, 5))

plt.plot(
    reinforce_rewards,
    label="REINFORCE"
)

plt.plot(
    a2c_rewards,
    label="A2C"
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Convergence")

plt.legend()
plt.grid()

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    reinforce_waiting,
    label="REINFORCE"
)

plt.plot(
    a2c_waiting,
    label="A2C"
)

plt.xlabel("Episode")
plt.ylabel("Average Waiting Vehicles")
plt.title("Vehicle Waiting Time")

plt.legend()
plt.grid()

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    reinforce_throughput,
    label="REINFORCE"
)

plt.plot(
    a2c_throughput,
    label="A2C"
)

plt.xlabel("Episode")
plt.ylabel("Vehicles Passed")
plt.title("Traffic Throughput")

plt.legend()
plt.grid()

plt.show()