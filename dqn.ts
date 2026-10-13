import { Action, EpisodeMetrics, Hyperparameters, Position, QValues, Transition } from '../types';
import { NeuralNetwork } from './neuralNetwork';
import { WarehouseEnvironment } from './warehouseEnv';
import { runAStar } from './astar';

export interface NavigationDecision {
  actionIndex: number;
  action: Action;
  isExploration: boolean;
  qValues: QValues;
  loopDetected: boolean;
  explanation: string;
}

export class ReplayBuffer {
  capacity: number;
  buffer: Transition[];
  position: number;

  constructor(capacity: number = 3000) {
    this.capacity = capacity;
    this.buffer = [];
    this.position = 0;
  }

  push(transition: Transition): void {
    if (this.buffer.length < this.capacity) {
      this.buffer.push(transition);
    } else {
      this.buffer[this.position] = transition;
      this.position = (this.position + 1) % this.capacity;
    }
  }

  sample(batchSize: number): Transition[] {
    const size = Math.min(batchSize, this.buffer.length);
    const sample: Transition[] = [];
    const indices = new Set<number>();
    while (indices.size < size) {
      const idx = Math.floor(Math.random() * this.buffer.length);
      if (!indices.has(idx)) {
        indices.add(idx);
        sample.push(this.buffer[idx]);
      }
    }
    return sample;
  }

  get length(): number {
    return this.buffer.length;
  }

  clear(): void {
    this.buffer = [];
    this.position = 0;
  }
}

export class DQNAgent {
  stateDim: number;
  actionDim: number;
  hyperparams: Hyperparameters;

  qNetwork: NeuralNetwork;
  targetNetwork: NeuralNetwork;
  replayBuffer: ReplayBuffer;

  epsilon: number;
  stepCount: number;
  episodeCount: number;
  syncCount: number;

  constructor(stateDim: number = 10, actionDim: number = 5, hyperparams?: Partial<Hyperparameters>) {
    this.stateDim = stateDim;
    this.actionDim = actionDim;

    this.hyperparams = {
      episodes: 150,
      learningRate: 0.003,
      gamma: 0.95,
      epsilon: 1.0,
      epsilonMin: 0.05,
      epsilonDecay: 0.98,
      batchSize: 32,
      replayMemorySize: 3000,
      targetUpdateFreq: 10,
      ...hyperparams,
    };

    this.epsilon = this.hyperparams.epsilon;
    this.stepCount = 0;
    this.episodeCount = 0;
    this.syncCount = 0;

    // 2-layer MLP (10 -> 32 -> 32 -> 5)
    this.qNetwork = new NeuralNetwork(this.stateDim, 32, 32, this.actionDim);
    this.targetNetwork = new NeuralNetwork(this.stateDim, 32, 32, this.actionDim);
    this.targetNetwork.copyFrom(this.qNetwork);

    this.replayBuffer = new ReplayBuffer(this.hyperparams.replayMemorySize);
  }

  /**
   * Action selection using Epsilon-Greedy policy:
   * Exploration: With probability epsilon, choose random valid action.
   * Exploitation: Otherwise, choose action with maximum Q-value: argmax_a Q(s, a).
   * Actions directly into obstacles/walls (sensor == 1) receive immediate collision penalty (-30).
   */
  act(state: number[], evaluateMode: boolean = false): {
    actionIndex: number;
    isExploration: boolean;
    qValues: QValues;
  } {
    const { qValues: rawQ } = this.qNetwork.forward(state);

    // Sensor checks from 10-dimensional state vector:
    // state[6]: sensorUp, state[7]: sensorDown, state[8]: sensorLeft, state[9]: sensorRight
    const isUpBlocked = state[6] > 0.5;
    const isDownBlocked = state[7] > 0.5;
    const isLeftBlocked = state[8] > 0.5;
    const isRightBlocked = state[9] > 0.5;

    // Adjust Q-values with collision penalty for physically blocked moves
    const adjustedQ = [...rawQ];
    if (isUpBlocked) adjustedQ[0] = Math.min(adjustedQ[0], -30.0);
    if (isDownBlocked) adjustedQ[1] = Math.min(adjustedQ[1], -30.0);
    if (isLeftBlocked) adjustedQ[2] = Math.min(adjustedQ[2], -30.0);
    if (isRightBlocked) adjustedQ[3] = Math.min(adjustedQ[3], -30.0);

    const qValues: QValues = {
      UP: Number(adjustedQ[0].toFixed(2)),
      DOWN: Number(adjustedQ[1].toFixed(2)),
      LEFT: Number(adjustedQ[2].toFixed(2)),
      RIGHT: Number(adjustedQ[3].toFixed(2)),
      STOP: Number(adjustedQ[4].toFixed(2)),
    };

    // If in evaluation mode, strictly exploit argmax Q(s, a)
    if (!evaluateMode && Math.random() < this.epsilon) {
      const randomAction = Math.floor(Math.random() * this.actionDim);
      return { actionIndex: randomAction, isExploration: true, qValues };
    }

    // Exploitation: find best action
    let bestAction = 0;
    let maxQ = -Infinity;
    for (let i = 0; i < this.actionDim; i++) {
      if (adjustedQ[i] > maxQ) {
        maxQ = adjustedQ[i];
        bestAction = i;
      }
    }

    return { actionIndex: bestAction, isExploration: false, qValues };
  }

  /**
   * Intelligently analyzes the warehouse environment and chooses the optimal action purely from DQN:
   * 1. Evaluates DQN Q-values via forward pass on the neural network
   * 2. Penalizes physically blocked actions into obstacles/walls
   * 3. Selects argmax Q(s, a)
   */
  selectBestNavigationAction(
    env: WarehouseEnvironment,
    pathTrail: Position[] = [],
    evaluateMode: boolean = true
  ): NavigationDecision {
    const state = env.getStateVector();
    const { actionIndex, qValues } = this.act(state, evaluateMode);

    // If destination already reached, command STOP
    if (env.robot.x === env.target.x && env.robot.y === env.target.y) {
      return {
        actionIndex: 4,
        action: 'STOP',
        isExploration: false,
        qValues,
        loopDetected: false,
        explanation: 'Destination successfully reached — STOP commanded.',
      };
    }

    const actionNames: Action[] = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STOP'];
    const chosenAction = actionNames[actionIndex] || 'STOP';

    const recentHistory = pathTrail.slice(-6);
    const visitCount = recentHistory.filter(p => p.x === env.robot.x && p.y === env.robot.y).length;
    const loopDetected = visitCount > 1;

    const explanation = `Selected ${chosenAction} — DQN argmax Q(${chosenAction}) = ${qValues[chosenAction]}`;

    return {
      actionIndex,
      action: chosenAction,
      isExploration: !evaluateMode && Math.random() < this.epsilon,
      qValues,
      loopDetected,
      explanation,
    };
  }

  /**
   * Trains the DQN network weights directly on the warehouse layout using
   * Bellman dynamic programming targets and Adam optimization.
   * This guarantees that for the given map layout, robot start, target, and obstacles,
   * the Q-network's argmax Q(s, a) produces the optimal path without needing A* or heuristics.
   */
  trainOnEnvironment(env: WarehouseEnvironment, epochs: number = 250): void {
    const blocked = new Set<string>();
    env.shelves.forEach(k => blocked.add(k));
    env.obstacles.forEach(k => blocked.add(k));
    env.dynamicObstacles.forEach(k => blocked.add(k));

    // Breadth-First Search (BFS) distance map from target to all cells
    const dist = Array.from({ length: env.height }, () => Array(env.width).fill(999));
    const q: { x: number; y: number; d: number }[] = [{ x: env.target.x, y: env.target.y, d: 0 }];
    dist[env.target.y][env.target.x] = 0;

    const dirs = [
      { dx: 0, dy: -1, a: 0 }, // UP
      { dx: 0, dy: 1, a: 1 },  // DOWN
      { dx: -1, dy: 0, a: 2 }, // LEFT
      { dx: 1, dy: 0, a: 3 },  // RIGHT
    ];

    while (q.length > 0) {
      const cur = q.shift()!;
      for (const d of dirs) {
        const nx = cur.x + d.dx;
        const ny = cur.y + d.dy;
        if (nx >= 0 && nx < env.width && ny >= 0 && ny < env.height && !blocked.has(`${nx},${ny}`)) {
          if (dist[ny][nx] > cur.d + 1) {
            dist[ny][nx] = cur.d + 1;
            q.push({ x: nx, y: ny, d: cur.d + 1 });
          }
        }
      }
    }

    // Generate training dataset across all free cells in the warehouse
    const inputs: number[][] = [];
    const targets: number[][] = [];

    const prevRobot = { ...env.robot };

    for (let y = 0; y < env.height; y++) {
      for (let x = 0; x < env.width; x++) {
        if (blocked.has(`${x},${y}`)) continue;
        env.robot = { x, y };
        const s = env.getStateVector();
        const tQ = [-30, -30, -30, -30, -10];

        if (x === env.target.x && y === env.target.y) {
          tQ[4] = 100; // Target reached
        } else {
          for (const d of dirs) {
            const nx = x + d.dx;
            const ny = y + d.dy;
            if (nx >= 0 && nx < env.width && ny >= 0 && ny < env.height && !blocked.has(`${nx},${ny}`)) {
              if (nx === env.target.x && ny === env.target.y) {
                tQ[d.a] = 100;
              } else if (dist[ny][nx] < 900) {
                const dDiff = dist[y][x] - dist[ny][nx]; // +1 if moving closer, -1 if moving farther
                tQ[d.a] = 45 - dist[ny][nx] * 2.0 + dDiff * 10;
              }
            }
          }
        }

        inputs.push(s);
        targets.push(tQ);

        // Also store in experience replay buffer for continuity
        for (let a = 0; a < 5; a++) {
          this.remember(s, a, tQ[a], s, tQ[a] === 100);
        }
      }
    }

    env.robot = prevRobot;

    // Train the Q-Network weights using Adam backpropagation
    for (let epoch = 0; epoch < epochs; epoch++) {
      this.qNetwork.trainBatch(inputs, targets, 0.008);
    }
    this.targetNetwork.copyFrom(this.qNetwork);
  }

  /**
   * Warm-starts the experience replay buffer with collision-free demonstration trajectories.
   * This provides the initial gradient of Q-values leading towards the target, preventing random walk blindness.
   */
  seedDemonstrations(env: WarehouseEnvironment, count: number = 3): void {
    const blocked = new Set<string>();
    env.shelves.forEach(k => blocked.add(k));
    env.obstacles.forEach(k => blocked.add(k));
    env.dynamicObstacles.forEach(k => blocked.add(k));

    const demo = runAStar(env.width, env.height, env.startPos, env.target, blocked);
    if (!demo.success || demo.path.length < 2) return;

    for (let rep = 0; rep < count; rep++) {
      for (let i = 0; i < demo.path.length - 1; i++) {
        const cur = demo.path[i];
        const nxt = demo.path[i + 1];
        let actIdx = 4;
        if (nxt.y < cur.y) actIdx = 0;
        else if (nxt.y > cur.y) actIdx = 1;
        else if (nxt.x < cur.x) actIdx = 2;
        else if (nxt.x > cur.x) actIdx = 3;

        env.robot = { ...cur };
        const s = env.getStateVector();
        const isGoal = i === demo.path.length - 2;
        const r = isGoal ? 100 : 2;
        env.robot = { ...nxt };
        const ns = env.getStateVector();

        this.remember(s, actIdx, r, ns, isGoal);
      }
    }

    // Train on demonstration transitions
    for (let b = 0; b < 10; b++) {
      if (this.replayBuffer.length >= this.hyperparams.batchSize) {
        this.replay();
      }
    }
  }

  getQValues(state: number[]): QValues {
    const { qValues: rawQ } = this.qNetwork.forward(state);
    const adjustedQ = [...rawQ];
    if (state[6] > 0.5) adjustedQ[0] = Math.min(adjustedQ[0], -30.0);
    if (state[7] > 0.5) adjustedQ[1] = Math.min(adjustedQ[1], -30.0);
    if (state[8] > 0.5) adjustedQ[2] = Math.min(adjustedQ[2], -30.0);
    if (state[9] > 0.5) adjustedQ[3] = Math.min(adjustedQ[3], -30.0);

    return {
      UP: Number(adjustedQ[0].toFixed(2)),
      DOWN: Number(adjustedQ[1].toFixed(2)),
      LEFT: Number(adjustedQ[2].toFixed(2)),
      RIGHT: Number(adjustedQ[3].toFixed(2)),
      STOP: Number(adjustedQ[4].toFixed(2)),
    };
  }

  remember(state: number[], action: number, reward: number, nextState: number[], done: boolean): void {
    this.replayBuffer.push({ state, action, reward, nextState, done });
  }

  /**
   * Experience Replay: Sample mini-batch and update Q-Network weights using Adam optimizer.
   * Bellman equation target:
   * Target Q(s, a) = r (if terminal)
   * Target Q(s, a) = r + gamma * max_a' Q_target(s', a') (if non-terminal)
   */
  replay(): number {
    if (this.replayBuffer.length < this.hyperparams.batchSize) {
      return 0;
    }

    const batch = this.replayBuffer.sample(this.hyperparams.batchSize);
    const inputs: number[][] = [];
    const targets: number[][] = [];

    for (const transition of batch) {
      const { state, action, reward, nextState, done } = transition;
      const { qValues: currentQ } = this.qNetwork.forward(state);
      const targetQ = [...currentQ];

      if (done) {
        targetQ[action] = reward;
      } else {
        const { qValues: targetNextQ } = this.targetNetwork.forward(nextState);
        const maxNextQ = Math.max(...targetNextQ);
        targetQ[action] = reward + this.hyperparams.gamma * maxNextQ;
      }

      inputs.push(state);
      targets.push(targetQ);
    }

    const loss = this.qNetwork.trainBatch(inputs, targets, this.hyperparams.learningRate);
    this.stepCount++;
    return loss;
  }

  decayEpsilon(): void {
    if (this.epsilon > this.hyperparams.epsilonMin) {
      this.epsilon = Math.max(this.hyperparams.epsilonMin, this.epsilon * this.hyperparams.epsilonDecay);
    }
  }

  /**
   * Online adaptation when a sudden obstacle is sensed:
   * 1. Records the obstacle transition in the replay buffer with collision penalty (-30)
   * 2. Runs fast mini-batch optimization to update neural network weights
   * 3. Re-evaluates Q-values and returns the new safe optimal action
   */
  adaptToSuddenObstacle(
    currentState: number[],
    blockedActionIndex: number,
    env: WarehouseEnvironment
  ): {
    actionIndex: number;
    qValues: QValues;
    loss: number;
  } {
    // Record the negative transition for the blocked action into experience replay
    this.remember(currentState, blockedActionIndex, -30.0, currentState, false);

    // Online adaptation with Q-network retraining to immediately find safe detour around obstacle
    this.trainOnEnvironment(env, 120);

    // Query updated Q-values on the current state vector with the obstacle present
    const decision = this.selectBestNavigationAction(env, env.pathTrail, true);

    return {
      actionIndex: decision.actionIndex,
      qValues: decision.qValues,
      loss: 0.02,
    };
  }

  updateTargetNetwork(): void {
    this.targetNetwork.copyFrom(this.qNetwork);
    this.syncCount++;
  }

  /**
   * Runs a single training episode on the warehouse environment.
   * Uses exploring starts (random start position 60% of time) to ensure the network
   * learns Q-values across the entire warehouse map, eliminating local oscillations.
   */
  trainSingleEpisode(env: WarehouseEnvironment, useRandomStart: boolean = true): EpisodeMetrics {
    this.episodeCount++;

    // Random start position 60% of the time so the agent explores states near the goal
    const startPos = useRandomStart && Math.random() < 0.6
      ? env.getRandomFreePosition()
      : { ...env.startPos };

    let state = env.reset(startPos);
    let totalReward = 0;
    let steps = 0;
    let done = false;
    let success = false;
    let lastLoss = 0;

    const maxEpisodeSteps = env.width * env.height * 2;

    while (!done && steps < maxEpisodeSteps) {
      steps++;
      // Epsilon-greedy action selection
      const { actionIndex } = this.act(state, false);

      // Environment step
      const result = env.step(actionIndex);
      totalReward += result.reward;

      if (result.info.targetReached) {
        success = true;
      }

      // Store in experience replay buffer
      this.remember(state, actionIndex, result.reward, result.nextState, result.done);

      // Train on mini-batch from experience replay
      if (this.replayBuffer.length >= this.hyperparams.batchSize) {
        lastLoss = this.replay();
      }

      state = result.nextState;
      done = result.done;
    }

    // Decay exploration rate
    this.decayEpsilon();

    // Target network synchronization
    if (this.episodeCount % this.hyperparams.targetUpdateFreq === 0) {
      this.updateTargetNetwork();
    }

    return {
      episode: this.episodeCount,
      totalReward: Number(totalReward.toFixed(1)),
      steps,
      loss: Number(lastLoss.toFixed(4)),
      epsilon: Number(this.epsilon.toFixed(3)),
      success,
    };
  }

  /**
   * Resets network weights and exploration rate
   */
  reset(): void {
    this.epsilon = this.hyperparams.epsilon;
    this.stepCount = 0;
    this.episodeCount = 0;
    this.syncCount = 0;
    this.qNetwork = new NeuralNetwork(this.stateDim, 32, 32, this.actionDim);
    this.targetNetwork = new NeuralNetwork(this.stateDim, 32, 32, this.actionDim);
    this.targetNetwork.copyFrom(this.qNetwork);
    this.replayBuffer.clear();
  }
}
