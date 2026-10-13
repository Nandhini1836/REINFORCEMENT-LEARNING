import { Action, ACTIONS, Position, RewardConfig, WarehouseConfig, WarehouseMapId, WarehouseMapPreset } from '../types';

export const WAREHOUSE_MAPS: WarehouseMapPreset[] = [
  { id: 'basic', name: 'Basic Warehouse', description: '2 vertical storage racks with cross-aisles (Default)' },
  { id: 'narrow', name: 'Narrow Aisles', description: '3 horizontal shelf rows with tight pass-through corridors' },
  { id: 'shelves', name: 'Multiple Shelves', description: '4 modular shelf blocks forming intersecting grid aisles' },
  { id: 'blocked', name: 'Blocked Path', description: 'Central obstacle wall forcing a detour around perimeter' },
  { id: 'complex', name: 'Complex Warehouse', description: 'Multi-aisle layout with scattered obstacles and mixed racks' },
];

export class WarehouseEnvironment {
  width: number;
  height: number;
  robot: Position;
  target: Position;
  startPos: Position;
  shelves: Set<string>;
  obstacles: Set<string>;
  dynamicObstacles: Set<string>;
  currentMapId: WarehouseMapId;
  rewardConfig: RewardConfig;
  maxSteps: number;
  currentStep: number;
  lastAction: Action | null;
  pathTrail: Position[];

  constructor(width: number = 10, height: number = 10) {
    this.width = width;
    this.height = height;
    this.startPos = { x: 1, y: 1 };
    this.robot = { ...this.startPos };
    this.target = { x: width - 2, y: height - 2 }; // e.g. (8, 8)
    this.shelves = new Set<string>();
    this.obstacles = new Set<string>();
    this.dynamicObstacles = new Set<string>();
    this.currentMapId = 'basic';
    this.lastAction = null;
    this.pathTrail = [{ ...this.robot }];
    this.maxSteps = width * height * 2;
    this.currentStep = 0;

    this.rewardConfig = {
      targetReached: 100,
      collision: -30,
      closerToTarget: 2,
      furtherFromTarget: -3,
      stepPenalty: -1,
      stopPenalty: -5,
      reversalPenalty: -3,
    };

    this.loadMap('basic');
  }

  /**
   * Loads one of the 5 predefined warehouse map configurations:
   * 1. Basic Warehouse (Default)
   * 2. Narrow Aisles
   * 3. Multiple Shelves
   * 4. Blocked Path
   * 5. Complex Warehouse
   */
  loadMap(mapId: WarehouseMapId): void {
    this.currentMapId = mapId;
    this.shelves.clear();
    this.obstacles.clear();
    this.dynamicObstacles.clear();

    switch (mapId) {
      case 'basic': {
        // Basic Warehouse: 2 vertical storage racks with central cross-aisles
        this.startPos = { x: 1, y: 1 };
        this.target = { x: this.width - 2, y: this.height - 2 };
        for (const x of [3, 6]) {
          for (let y = 1; y <= this.height - 2; y++) {
            if (y !== 4 && y !== 5) {
              this.shelves.add(`${x},${y}`);
            }
          }
        }
        this.obstacles.add('2,6');
        this.obstacles.add('7,3');
        break;
      }

      case 'narrow': {
        // Narrow Aisles: 3 horizontal shelf rows with tight pass-through corridors
        this.startPos = { x: 1, y: 1 };
        this.target = { x: 8, y: 9 };
        for (const y of [2, 5, 8]) {
          for (let x = 1; x <= this.width - 2; x++) {
            // Aisle gaps
            if (x !== 2 && x !== 7) {
              this.shelves.add(`${x},${y}`);
            }
          }
        }
        this.obstacles.add('4,1');
        this.obstacles.add('5,6');
        break;
      }

      case 'shelves': {
        // Multiple Shelves: 4 separate storage blocks forming grid corridors
        this.startPos = { x: 0, y: 1 };
        this.target = { x: 9, y: 8 };
        // Storage block 1: (2..3, 1..3)
        for (let x = 2; x <= 3; x++) {
          for (let y = 1; y <= 3; y++) this.shelves.add(`${x},${y}`);
        }
        // Storage block 2: (6..7, 1..3)
        for (let x = 6; x <= 7; x++) {
          for (let y = 1; y <= 3; y++) this.shelves.add(`${x},${y}`);
        }
        // Storage block 3: (2..3, 6..8)
        for (let x = 2; x <= 3; x++) {
          for (let y = 6; y <= 8; y++) this.shelves.add(`${x},${y}`);
        }
        // Storage block 4: (6..7, 6..8)
        for (let x = 6; x <= 7; x++) {
          for (let y = 6; y <= 8; y++) this.shelves.add(`${x},${y}`);
        }
        this.obstacles.add('4,4');
        this.obstacles.add('5,5');
        break;
      }

      case 'blocked': {
        // Blocked Path: Direct corridor is blocked by central barrier, forcing perimeter detour
        this.startPos = { x: 1, y: 5 };
        this.target = { x: 8, y: 5 };
        // Central vertical barricade at x=4 from y=2 to y=7
        for (let y = 2; y <= 7; y++) {
          this.shelves.add(`4,${y}`);
        }
        // Supporting shelf units
        this.shelves.add('2,2');
        this.shelves.add('2,3');
        this.shelves.add('6,6');
        this.shelves.add('6,7');
        this.obstacles.add('3,5');
        this.obstacles.add('5,5');
        break;
      }

      case 'complex': {
        // Complex Warehouse: Multi-zone layout with pallet staging, storage racks, and obstacles
        this.startPos = { x: 1, y: 1 };
        this.target = { x: 8, y: 8 };
        // Racks
        for (let y = 1; y <= 4; y++) this.shelves.add(`3,${y}`);
        for (let y = 5; y <= 8; y++) this.shelves.add(`6,${y}`);
        for (let x = 4; x <= 5; x++) this.shelves.add(`${x},3`);
        for (let x = 4; x <= 5; x++) this.shelves.add(`${x},6`);
        this.obstacles.add('1,4');
        this.obstacles.add('8,5');
        this.obstacles.add('4,8');
        break;
      }
    }

    this.reset();
  }

  loadDefaultLayout(): void {
    this.loadMap('basic');
  }

  isBlocked(x: number, y: number): boolean {
    // Boundary check
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) return true;
    const key = `${x},${y}`;
    return this.shelves.has(key) || this.obstacles.has(key) || this.dynamicObstacles.has(key);
  }

  getManhattanDistance(p1: Position, p2: Position): number {
    return Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y);
  }

  getEuclideanDistance(p1: Position, p2: Position): number {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
  }

  /**
   * MODULE 1: State Representation (10 dimensions)
   * Represents:
   * 1. Robot's current position (x, y)
   * 2. Destination position (targetX, targetY)
   * 3. Direction delta to destination (dx, dy)
   * 4. Nearby obstacles (UP, DOWN, LEFT, RIGHT free/blocked)
   */
  getStateVector(pos: Position = this.robot): number[] {
    const normX = pos.x / (this.width - 1);
    const normY = pos.y / (this.height - 1);
    const targetNormX = this.target.x / (this.width - 1);
    const targetNormY = this.target.y / (this.height - 1);

    const dx = (this.target.x - pos.x) / (this.width - 1);
    const dy = (this.target.y - pos.y) / (this.height - 1);

    // 1-step obstacle sensors (1 = blocked/boundary, 0 = free)
    const sensorUp = this.isBlocked(pos.x, pos.y - 1) ? 1.0 : 0.0;
    const sensorDown = this.isBlocked(pos.x, pos.y + 1) ? 1.0 : 0.0;
    const sensorLeft = this.isBlocked(pos.x - 1, pos.y) ? 1.0 : 0.0;
    const sensorRight = this.isBlocked(pos.x + 1, pos.y) ? 1.0 : 0.0;

    return [
      normX,
      normY,
      targetNormX,
      targetNormY,
      dx,
      dy,
      sensorUp,
      sensorDown,
      sensorLeft,
      sensorRight,
    ];
  }

  /**
   * Executes an action in the environment and returns the reward, next state, and done flag.
   */
  step(actionIndex: number): {
    nextState: number[];
    reward: number;
    done: boolean;
    info: {
      action: Action;
      targetReached: boolean;
      collision: boolean;
      timeout: boolean;
      prevPos: Position;
      newPos: Position;
    };
  } {
    this.currentStep++;
    const action = ACTIONS[actionIndex] || 'STOP';
    const prevPos = { ...this.robot };
    const prevDist = this.getManhattanDistance(this.robot, this.target);

    let nextPos = { ...this.robot };
    if (action === 'UP') nextPos.y -= 1;
    else if (action === 'DOWN') nextPos.y += 1;
    else if (action === 'LEFT') nextPos.x -= 1;
    else if (action === 'RIGHT') nextPos.x += 1;

    let reward = 0;
    let done = false;
    let collision = false;
    let targetReached = false;
    let timeout = false;

    // Check collision with boundaries, shelves, or static obstacles
    if (action !== 'STOP' && this.isBlocked(nextPos.x, nextPos.y)) {
      collision = true;
      reward = this.rewardConfig.collision;
      done = true; // collision ends the trial
    } else if (action === 'STOP') {
      reward = this.rewardConfig.stopPenalty;
    } else {
      // Valid move
      this.robot = nextPos;
      this.pathTrail.push({ ...this.robot });

      // Check if target reached
      if (this.robot.x === this.target.x && this.robot.y === this.target.y) {
        reward = this.rewardConfig.targetReached;
        done = true;
        targetReached = true;
      } else {
        const newDist = this.getManhattanDistance(this.robot, this.target);

        // Distance progress reward
        if (newDist < prevDist) {
          reward = this.rewardConfig.closerToTarget + this.rewardConfig.stepPenalty;
        } else {
          reward = this.rewardConfig.furtherFromTarget + this.rewardConfig.stepPenalty;
        }

        // Anti-oscillation penalty: if immediately reversing previous move
        const isReversal =
          (this.lastAction === 'LEFT' && action === 'RIGHT') ||
          (this.lastAction === 'RIGHT' && action === 'LEFT') ||
          (this.lastAction === 'UP' && action === 'DOWN') ||
          (this.lastAction === 'DOWN' && action === 'UP');

        if (isReversal) {
          reward += this.rewardConfig.reversalPenalty;
        }

        // Timeout check
        if (this.currentStep >= this.maxSteps) {
          done = true;
          timeout = true;
        }
      }
    }

    this.lastAction = action;
    const nextState = this.getStateVector();

    return {
      nextState,
      reward,
      done,
      info: {
        action,
        targetReached,
        collision,
        timeout,
        prevPos,
        newPos: { ...this.robot },
      },
    };
  }

  reset(tempStartPos?: Position, tempTargetPos?: Position, clearDynamic: boolean = false, persistStart: boolean = false): number[] {
    this.currentStep = 0;
    this.lastAction = null;
    if (clearDynamic) {
      this.dynamicObstacles.clear();
    }
    if (persistStart && tempStartPos) {
      this.startPos = { ...tempStartPos };
    }
    if (tempTargetPos) {
      this.target = { ...tempTargetPos };
    }
    this.robot = tempStartPos ? { ...tempStartPos } : { ...this.startPos };
    this.pathTrail = [{ ...this.robot }];
    return this.getStateVector();
  }

  setStartPos(pos: Position): void {
    this.startPos = { ...pos };
    this.robot = { ...pos };
    this.pathTrail = [{ ...this.robot }];
  }

  setTargetPos(pos: Position): void {
    this.target = { ...pos };
  }

  clearTrail(): void {
    this.pathTrail = [{ ...this.robot }];
    this.currentStep = 0;
  }

  // --- DYNAMIC OBSTACLE METHODS (SPECIAL INNOVATIVE FEATURE) ---

  /**
   * Adds a dynamic obstacle at (x, y).
   * Ensures the cell is within boundaries and does not overlap with the robot's current position or target.
   */
  addDynamicObstacle(x: number, y: number): boolean {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) return false;
    if ((x === this.robot.x && y === this.robot.y) || (x === this.target.x && y === this.target.y)) return false;

    const key = `${x},${y}`;
    // Clear any static shelf or obstacle on this cell so it acts as dynamic
    this.shelves.delete(key);
    this.obstacles.delete(key);
    this.dynamicObstacles.add(key);
    return true;
  }

  removeDynamicObstacle(x: number, y: number): boolean {
    const key = `${x},${y}`;
    return this.dynamicObstacles.delete(key);
  }

  toggleDynamicObstacle(x: number, y: number): boolean {
    const key = `${x},${y}`;
    if ((x === this.robot.x && y === this.robot.y) || (x === this.target.x && y === this.target.y)) return false;

    if (this.dynamicObstacles.has(key)) {
      this.dynamicObstacles.delete(key);
      return false;
    } else {
      this.shelves.delete(key);
      this.obstacles.delete(key);
      this.dynamicObstacles.add(key);
      return true;
    }
  }

  clearDynamicObstacles(): void {
    this.dynamicObstacles.clear();
  }

  /**
   * Finds the cell directly in front of the robot based on its direction or heading toward the target.
   * Useful for the "Insert in Robot Path" one-click innovation demo!
   */
  getFrontCellInRobotPath(heading?: Action | null): Position | null {
    const r = this.robot;
    const t = this.target;

    // First try the robot's current heading direction if specified
    if (heading && heading !== 'STOP') {
      const delta: Record<Action, { dx: number; dy: number }> = {
        UP: { dx: 0, dy: -1 },
        DOWN: { dx: 0, dy: 1 },
        LEFT: { dx: -1, dy: 0 },
        RIGHT: { dx: 1, dy: 0 },
        STOP: { dx: 0, dy: 0 },
      };
      const candX = r.x + delta[heading].dx;
      const candY = r.y + delta[heading].dy;
      if (
        candX >= 0 && candX < this.width &&
        candY >= 0 && candY < this.height &&
        !(candX === t.x && candY === t.y) &&
        !this.isBlocked(candX, candY)
      ) {
        return { x: candX, y: candY };
      }
    }

    // Otherwise determine primary vector toward target
    const dx = t.x - r.x;
    const dy = t.y - r.y;

    const candidates: Position[] = [];
    if (Math.abs(dx) >= Math.abs(dy)) {
      if (dx > 0) candidates.push({ x: r.x + 1, y: r.y });
      if (dx < 0) candidates.push({ x: r.x - 1, y: r.y });
      if (dy > 0) candidates.push({ x: r.x, y: r.y + 1 });
      if (dy < 0) candidates.push({ x: r.x, y: r.y - 1 });
    } else {
      if (dy > 0) candidates.push({ x: r.x, y: r.y + 1 });
      if (dy < 0) candidates.push({ x: r.x, y: r.y - 1 });
      if (dx > 0) candidates.push({ x: r.x + 1, y: r.y });
      if (dx < 0) candidates.push({ x: r.x - 1, y: r.y });
    }

    for (const cand of candidates) {
      if (
        cand.x >= 0 && cand.x < this.width &&
        cand.y >= 0 && cand.y < this.height &&
        !(cand.x === t.x && cand.y === t.y) &&
        !this.isBlocked(cand.x, cand.y)
      ) {
        return cand;
      }
    }

    return null;
  }

  setRobotStart(x: number, y: number): boolean {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) return false;
    if (x === this.target.x && y === this.target.y) return false;
    const key = `${x},${y}`;
    if (this.shelves.has(key) || this.obstacles.has(key) || this.dynamicObstacles.has(key)) {
      return false;
    }

    this.startPos = { x, y };
    this.reset();
    return true;
  }

  setTarget(x: number, y: number): boolean {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) return false;
    if (x === this.startPos.x && y === this.startPos.y) return false;
    const key = `${x},${y}`;
    if (this.shelves.has(key) || this.obstacles.has(key) || this.dynamicObstacles.has(key)) {
      return false;
    }

    this.target = { x, y };
    this.reset();
    return true;
  }

  setShelf(x: number, y: number): boolean {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) return false;
    if ((x === this.startPos.x && y === this.startPos.y) || (x === this.target.x && y === this.target.y)) return false;

    const key = `${x},${y}`;
    this.obstacles.delete(key);
    this.dynamicObstacles.delete(key);
    if (this.shelves.has(key)) {
      this.shelves.delete(key);
    } else {
      this.shelves.add(key);
    }
    return true;
  }

  setObstacle(x: number, y: number): boolean {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) return false;
    if ((x === this.startPos.x && y === this.startPos.y) || (x === this.target.x && y === this.target.y)) return false;

    const key = `${x},${y}`;
    this.shelves.delete(key);
    this.dynamicObstacles.delete(key);
    if (this.obstacles.has(key)) {
      this.obstacles.delete(key);
    } else {
      this.obstacles.add(key);
    }
    return true;
  }

  clearCell(x: number, y: number): void {
    const key = `${x},${y}`;
    this.shelves.delete(key);
    this.obstacles.delete(key);
    this.dynamicObstacles.delete(key);
  }

  clearAllObstacles(): void {
    this.shelves.clear();
    this.obstacles.clear();
    this.dynamicObstacles.clear();
    this.reset();
  }

  loadOpenLayout(): void {
    this.clearAllObstacles();
    this.startPos = { x: 1, y: 1 };
    this.target = { x: this.width - 2, y: this.height - 2 };
    this.reset();
  }

  loadNarrowAislesLayout(): void {
    this.shelves.clear();
    this.obstacles.clear();

    // 3 parallel horizontal shelf rows with aisle gaps
    for (const y of [2, 5, 8]) {
      for (let x = 1; x <= this.width - 2; x++) {
        if (x !== 3 && x !== 6) {
          this.shelves.add(`${x},${y}`);
        }
      }
    }
    this.obstacles.add('4,1');
    this.obstacles.add('5,7');
    this.startPos = { x: 0, y: 0 };
    this.target = { x: this.width - 1, y: this.height - 1 };
    this.reset();
  }

  toggleCell(x: number, y: number): 'empty' | 'shelf' | 'obstacle' {
    const key = `${x},${y}`;
    if ((x === this.startPos.x && y === this.startPos.y) || (x === this.target.x && y === this.target.y)) {
      return 'empty';
    }

    if (this.shelves.has(key)) {
      this.shelves.delete(key);
      this.obstacles.add(key);
      return 'obstacle';
    } else if (this.obstacles.has(key)) {
      this.obstacles.delete(key);
      return 'empty';
    } else {
      this.shelves.add(key);
      return 'shelf';
    }
  }

  getRandomFreePosition(): Position {
    let attempts = 0;
    while (attempts < 200) {
      attempts++;
      const rx = Math.floor(Math.random() * this.width);
      const ry = Math.floor(Math.random() * this.height);
      if (!this.isBlocked(rx, ry) && !(rx === this.target.x && ry === this.target.y)) {
        return { x: rx, y: ry };
      }
    }
    return { x: this.startPos.x, y: this.startPos.y };
  }
}
