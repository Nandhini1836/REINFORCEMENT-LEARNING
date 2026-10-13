import { AlgorithmRunResult, Position } from '../types';

interface AStarNode {
  x: number;
  y: number;
  g: number;
  h: number;
  f: number;
  parent: AStarNode | null;
}

export function runAStar(
  gridWidth: number,
  gridHeight: number,
  start: Position,
  target: Position,
  blockedSet: Set<string>
): AlgorithmRunResult {
  const startKey = `${start.x},${start.y}`;
  const targetKey = `${target.x},${target.y}`;

  if (blockedSet.has(startKey) || blockedSet.has(targetKey)) {
    return {
      algorithm: 'A*',
      success: false,
      pathLength: 0,
      steps: 0,
      collisionCount: 1,
      totalReward: -30,
      path: [start],
    };
  }

  const heuristic = (p1: Position, p2: Position): number => {
    return Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y);
  };

  const openList: AStarNode[] = [];
  const closedSet = new Set<string>();

  const startNode: AStarNode = {
    x: start.x,
    y: start.y,
    g: 0,
    h: heuristic(start, target),
    f: heuristic(start, target),
    parent: null,
  };

  openList.push(startNode);
  let nodesExpanded = 0;

  const neighbors = [
    { dx: 0, dy: -1 }, // UP
    { dx: 0, dy: 1 },  // DOWN
    { dx: -1, dy: 0 }, // LEFT
    { dx: 1, dy: 0 },  // RIGHT
  ];

  while (openList.length > 0) {
    let bestIdx = 0;
    for (let i = 1; i < openList.length; i++) {
      if (openList[i].f < openList[bestIdx].f) {
        bestIdx = i;
      }
    }

    const current = openList.splice(bestIdx, 1)[0];
    const currentKey = `${current.x},${current.y}`;
    nodesExpanded++;

    if (current.x === target.x && current.y === target.y) {
      const path: Position[] = [];
      let curr: AStarNode | null = current;
      while (curr !== null) {
        path.unshift({ x: curr.x, y: curr.y });
        curr = curr.parent;
      }

      const pathLength = path.length - 1;
      return {
        algorithm: 'A*',
        success: true,
        pathLength,
        steps: pathLength,
        collisionCount: 0,
        totalReward: Math.max(10, 100 - pathLength * 2),
        path,
      };
    }

    closedSet.add(currentKey);

    for (const offset of neighbors) {
      const nx = current.x + offset.dx;
      const ny = current.y + offset.dy;
      const neighborKey = `${nx},${ny}`;

      if (nx < 0 || nx >= gridWidth || ny < 0 || ny >= gridHeight) {
        continue;
      }

      if (blockedSet.has(neighborKey) || closedSet.has(neighborKey)) {
        continue;
      }

      const tentativeG = current.g + 1;
      const existingOpen = openList.find(n => n.x === nx && n.y === ny);

      if (!existingOpen) {
        const h = heuristic({ x: nx, y: ny }, target);
        openList.push({
          x: nx,
          y: ny,
          g: tentativeG,
          h,
          f: tentativeG + h,
          parent: current,
        });
      } else if (tentativeG < existingOpen.g) {
        existingOpen.g = tentativeG;
        existingOpen.f = tentativeG + existingOpen.h;
        existingOpen.parent = current;
      }
    }
  }

  return {
    algorithm: 'A*',
    success: false,
    pathLength: 0,
    steps: nodesExpanded,
    collisionCount: 0,
    totalReward: -50,
    path: [start],
  };
}
