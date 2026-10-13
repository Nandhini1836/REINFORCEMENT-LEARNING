import numpy as np

class TrafficEnvironment:
    def __init__(self, agents=4):
        self.agents = agents
        self.queues = np.zeros((agents, 4), dtype=int)

    def reset(self):
        self.queues = np.random.randint(1, 10, (self.agents, 4))
        return self.queues.copy()

    def step(self, actions):
        throughput = 0

        for i in range(self.agents):
            lanes = [0, 1] if actions[i] == 0 else [2, 3]

            for lane in lanes:
                passed = min(self.queues[i][lane], 3)
                self.queues[i][lane] -= passed
                throughput += passed

            self.queues[i] += np.random.randint(0, 3, 4)

        waiting = np.sum(self.queues)
        reward = -waiting

        return self.queues.copy(), reward, throughput


class TrafficAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def select_action(self, state):
        ns = state[0] + state[1]
        ew = state[2] + state[3]

        return 0 if ns > ew else 1


# Create environment
env = TrafficEnvironment(agents=4)

# Create agents
agents = [TrafficAgent(i) for i in range(4)]

# Start simulation
state = env.reset()

print("===================================")
print(" ADAPTIVE TRAFFIC SIGNAL CONTROL")
print("===================================")
print("\nInitial Traffic State:")
print(state)

for step in range(10):

    actions = [
        agents[i].select_action(state[i])
        for i in range(4)
    ]

    state, reward, throughput = env.step(actions)

    print("\nStep:", step + 1)
    print("Agent Actions:", actions)
    print("Reward:", reward)
    print("Throughput:", throughput)

print("\nTraffic Optimization Completed!")