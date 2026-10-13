export type Action = 'UP' | 'DOWN' | 'LEFT' | 'RIGHT' | 'STOP';
export const ACTIONS: Action[] = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STOP'];

export interface Position {
  x: number;
  y: number;
}

export interface QValues {
  UP: number;
  DOWN: number;
  LEFT: number;
  RIGHT: number;
  STOP: number;
}

export interface RewardConfig {
  targetReached: number;
  collision: number;
  closerToTarget: number;
  furtherFromTarget: number;
  stepPenalty: number;
  stopPenalty: number;
  reversalPenalty: number;
}

export interface Hyperparameters {
  episodes: number;
  learningRate: number;
  gamma: number;
  epsilon: number;
  epsilonMin: number;
  epsilonDecay: number;
  batchSize: number;
  replayMemorySize: number;
  targetUpdateFreq: number;
}

export interface Transition {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

export interface EpisodeMetrics {
  episode: number;
  totalReward: number;
  steps: number;
  loss: number;
  epsilon: number;
  success: boolean;
}

export interface AlgorithmRunResult {
  algorithm: 'A*' | 'DQN';
  success: boolean;
  pathLength: number;
  steps: number;
  collisionCount: number;
  totalReward: number;
  path: Position[];
}

export interface WarehouseConfig {
  width: number;
  height: number;
  robotStart: Position;
  target: Position;
  shelves: Position[];
  obstacles: Position[];
}

export type WarehouseMapId = 'basic' | 'narrow' | 'shelves' | 'blocked' | 'complex';

export interface WarehouseMapPreset {
  id: WarehouseMapId;
  name: string;
  description: string;
}

export type EditTool = 'ROBOT' | 'TARGET' | 'SHELF' | 'OBSTACLE' | 'DYNAMIC_OBSTACLE' | 'ERASER';
