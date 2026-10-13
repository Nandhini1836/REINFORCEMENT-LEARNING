import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import { Action, ACTIONS, AlgorithmRunResult, EditTool, EpisodeMetrics, Position, QValues, WarehouseMapId } from './types';
import { WarehouseEnvironment, WAREHOUSE_MAPS } from './algorithms/warehouseEnv';
import { DQNAgent } from './algorithms/dqn';
import { runAStar } from './algorithms/astar';
import { WarehouseGrid } from './components/WarehouseGrid';
import {
  Play,
  Pause,
  RotateCcw,
  SkipForward,
  Cpu,
  Compass,
  GraduationCap,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Flame,
  Layers,
  ArrowRight,
  TrendingUp,
  Wrench,
  Eraser,
  MousePointerClick,
  Zap,
  MapPin,
  SlidersHorizontal,
  Trash2,
  Plus,
} from 'lucide-react';

export default function App() {
  // 1. Module 1: Warehouse Environment (10x10)
  const env = useMemo(() => new WarehouseEnvironment(10, 10), []);

  // 2. Module 2: DQN Agent (10 state inputs, 5 actions)
  const agent = useMemo(() => new DQNAgent(10, 5), []);

  // Simulation states
  const [algorithm, setAlgorithm] = useState<'DQN' | 'A*'>('DQN');
  const [selectedMap, setSelectedMap] = useState<WarehouseMapId>('basic');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [robotPos, setRobotPos] = useState<Position>({ ...env.robot });
  const [currentAction, setCurrentAction] = useState<Action | null>(null);
  const [stepReward, setStepReward] = useState<number>(0);
  const [accumulatedReward, setAccumulatedReward] = useState<number>(0);
  const [stepCount, setStepCount] = useState<number>(0);
  const [status, setStatus] = useState<'Ready' | 'Navigating' | 'Goal Reached' | 'Collision'>('Ready');
  const [qValues, setQValues] = useState<QValues>({ UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0, STOP: 0 });
  const [pathTrail, setPathTrail] = useState<Position[]>([{ ...env.robot }]);
  const [plannedPath, setPlannedPath] = useState<Position[]>([]);

  // Performance benchmark records (actual calculated values)
  const [dqnResult, setDqnResult] = useState<AlgorithmRunResult | null>(null);
  const [astarResult, setAstarResult] = useState<AlgorithmRunResult | null>(null);

  // Training state
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [showTrainingPanel, setShowTrainingPanel] = useState<boolean>(false);
  const [trainingHistory, setTrainingHistory] = useState<EpisodeMetrics[]>([]);
  const [isTrained, setIsTrained] = useState<boolean>(false);

  // Grid interactive placement states (Robot, Goal, Shelves, Obstacles, Dynamic Obstacles)
  const [activeTool, setActiveTool] = useState<EditTool>('ROBOT');
  const [showCustomizeToolbar, setShowCustomizeToolbar] = useState<boolean>(false);
  const [feedbackMessage, setFeedbackMessage] = useState<{
    text: string;
    type: 'error' | 'success' | 'info';
  } | null>(null);
  const [startPos, setStartPos] = useState<Position>({ ...env.startPos });
  const [targetPos, setTargetPos] = useState<Position>({ ...env.target });
  const [shelves, setShelves] = useState<Set<string>>(new Set(env.shelves));
  const [obstacles, setObstacles] = useState<Set<string>>(new Set(env.obstacles));
  const [dynamicObstacles, setDynamicObstacles] = useState<Set<string>>(new Set(env.dynamicObstacles));

  // Path Analysis & Intelligent Reasoning State
  const [navExplanation, setNavExplanation] = useState<string>('Environment analyzed. DQN policy primed and path analysis active.');
  const [loopDetectedNotice, setLoopDetectedNotice] = useState<boolean>(false);

  // Dynamic Obstacle Real-Time Adaptation Notification
  const [obstacleAlert, setObstacleAlert] = useState<{
    show: boolean;
    pos: Position;
    prevAction: Action;
    newAction: Action;
    phase?: 'STOPPED' | 'LEARNING' | 'RESUMING';
    timestamp: number;
  } | null>(null);

  // Running interval timer ref
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Blocked cells lookup for A*
  const blockedSet = useMemo(() => {
    const set = new Set<string>();
    shelves.forEach(k => set.add(k));
    obstacles.forEach(k => set.add(k));
    dynamicObstacles.forEach(k => set.add(k));
    return set;
  }, [shelves, obstacles, dynamicObstacles]);

  // Sudden Obstacle Sensing, Stopping, Learning, and Resuming to Destination
  const triggerSuddenObstacleReaction = (
    obstaclePos: Position,
    prevAct: Action,
    blockedActionIndex: number
  ) => {
    // 1. SENSE & STOP: Robot senses the new sudden obstacle and immediately stops
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRunning(false);
    setCurrentAction('STOP');
    setStatus('Alert / Stopped');
    setObstacleAlert({
      show: true,
      pos: obstaclePos,
      prevAction: prevAct,
      newAction: 'STOP',
      phase: 'STOPPED',
      timestamp: Date.now(),
    });

    // 2. LEARN: Robot trains on the collision risk and updates Q-network weights
    setTimeout(() => {
      const currentState = env.getStateVector();
      const adaptationResult = agent.adaptToSuddenObstacle(currentState, blockedActionIndex, env);
      setQValues(adaptationResult.qValues);
      const learnedAct = ACTIONS[adaptationResult.actionIndex];

      setObstacleAlert({
        show: true,
        pos: obstaclePos,
        prevAction: prevAct,
        newAction: learnedAct,
        phase: 'LEARNING',
        timestamp: Date.now(),
      });

      // 3. IMMEDIATELY START MOVING TO DESTINATION
      setTimeout(() => {
        setCurrentAction(learnedAct);
        setStatus('Navigating');
        setObstacleAlert({
          show: true,
          pos: obstaclePos,
          prevAction: prevAct,
          newAction: learnedAct,
          phase: 'RESUMING',
          timestamp: Date.now(),
        });
        setIsRunning(true);
      }, 400);
    }, 380);
  };

  // Initial training warm-up on mount (60 episodes + demonstration replay) so DQN works immediately out of the box!
  useEffect(() => {
    agent.seedDemonstrations(env, 3);
    const initialHistory: EpisodeMetrics[] = [];
    for (let i = 0; i < 60; i++) {
      const metric = agent.trainSingleEpisode(env, true);
      initialHistory.push(metric);
    }
    setTrainingHistory(initialHistory);
    setIsTrained(true);
    env.reset();
    setRobotPos({ ...env.robot });
    setPathTrail([{ ...env.robot }]);
    setQValues(agent.getQValues(env.getStateVector()));
    setNavExplanation('Environment analyzed. DQN policy primed and path analysis active.');
  }, [env, agent]);

  // Calculate planned path preview (A* or DQN projected path)
  useEffect(() => {
    if (algorithm === 'A*') {
      const res = runAStar(env.width, env.height, robotPos, env.target, blockedSet);
      setPlannedPath(res.path);
    } else {
      // Simulate DQN forward path projection from current robot position to target
      const simEnv = new WarehouseEnvironment(env.width, env.height);
      simEnv.shelves = new Set(env.shelves);
      simEnv.obstacles = new Set(env.obstacles);
      simEnv.dynamicObstacles = new Set(env.dynamicObstacles);
      simEnv.startPos = { ...robotPos };
      simEnv.target = { ...env.target };
      simEnv.robot = { ...robotPos };

      const projected: Position[] = [{ ...robotPos }];
      let simSteps = 0;
      while (simSteps < 25) {
        simSteps++;
        if (simEnv.robot.x === simEnv.target.x && simEnv.robot.y === simEnv.target.y) break;
        const dec = agent.selectBestNavigationAction(simEnv, projected, true);
        if (dec.action === 'STOP') break;
        const r = simEnv.step(dec.actionIndex);
        projected.push({ ...simEnv.robot });
        if (r.info.targetReached || r.info.collision) break;
      }
      setPlannedPath(projected.length > 1 ? projected : []);
    }
  }, [algorithm, env, blockedSet, robotPos, isTrained, selectedMap]);

  // Stop auto run on unmount or finish
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // Single Step execution
  const handleStep = () => {
    if (status === 'Goal Reached' || status === 'Collision') return;

    if (algorithm === 'A*') {
      // Step along A* planned path
      const astarRes = runAStar(env.width, env.height, robotPos, env.target, blockedSet);
      if (astarRes.path.length <= 1) {
        if (robotPos.x === env.target.x && robotPos.y === env.target.y) {
          setStatus('Goal Reached');
        } else {
          setStatus('Collision');
        }
        setIsRunning(false);
        return;
      }

      const nextPos = astarRes.path[1];
      let act: Action = 'STOP';
      if (nextPos.x > robotPos.x) act = 'RIGHT';
      else if (nextPos.x < robotPos.x) act = 'LEFT';
      else if (nextPos.y > robotPos.y) act = 'DOWN';
      else if (nextPos.y < robotPos.y) act = 'UP';

      // Move robot
      env.robot = { ...nextPos };
      env.pathTrail.push({ ...nextPos });
      const newStep = stepCount + 1;
      setRobotPos({ ...nextPos });
      setPathTrail([...env.pathTrail]);
      setCurrentAction(act);
      setStepCount(newStep);
      setStatus('Navigating');

      const dist = env.getManhattanDistance(nextPos, env.target);
      const rew = dist === 0 ? 100 : -1;
      setStepReward(rew);
      setAccumulatedReward(prev => prev + rew);

      if (nextPos.x === env.target.x && nextPos.y === env.target.y) {
        setStatus('Goal Reached');
        setIsRunning(false);
        setAstarResult({
          algorithm: 'A*',
          success: true,
          pathLength: env.pathTrail.length - 1,
          steps: newStep,
          collisionCount: 0,
          totalReward: accumulatedReward + rew,
          path: [...env.pathTrail],
        });
      }
    } else {
      // Step using DQN Policy: argmax Q(s, a)
      // Check if robot's current moving direction faces a sudden dynamic obstacle
      if (currentAction && currentAction !== 'STOP') {
        const front = env.getFrontCellInRobotPath(currentAction);
        if (front && env.dynamicObstacles.has(`${front.x},${front.y}`)) {
          let blockedIdx = 3;
          if (currentAction === 'UP') blockedIdx = 0;
          else if (currentAction === 'DOWN') blockedIdx = 1;
          else if (currentAction === 'LEFT') blockedIdx = 2;
          else if (currentAction === 'RIGHT') blockedIdx = 3;
          triggerSuddenObstacleReaction(front, currentAction, blockedIdx);
          return;
        }
      }

      // Intelligent DQN path analysis with loop avoidance & goal-seeking heuristic
      const navDecision = agent.selectBestNavigationAction(env, pathTrail, true);
      setQValues(navDecision.qValues);
      setCurrentAction(navDecision.action);
      setNavExplanation(navDecision.explanation);
      setLoopDetectedNotice(navDecision.loopDetected);

      if (navDecision.action === 'STOP' && env.robot.x === env.target.x && env.robot.y === env.target.y) {
        setStatus('Goal Reached');
        setIsRunning(false);
        setDqnResult({
          algorithm: 'DQN',
          success: true,
          pathLength: env.pathTrail.length - 1,
          steps: stepCount,
          collisionCount: 0,
          totalReward: accumulatedReward + 100,
          path: [...env.pathTrail],
        });
        return;
      }

      const stepRes = env.step(navDecision.actionIndex);
      const newStep = stepCount + 1;
      setRobotPos({ ...env.robot });
      setPathTrail([...env.pathTrail]);
      setStepReward(stepRes.reward);
      setAccumulatedReward(prev => prev + stepRes.reward);
      setStepCount(newStep);

      if (stepRes.info.targetReached) {
        setStatus('Goal Reached');
        setIsRunning(false);
        setDqnResult({
          algorithm: 'DQN',
          success: true,
          pathLength: env.pathTrail.length - 1,
          steps: newStep,
          collisionCount: 0,
          totalReward: accumulatedReward + stepRes.reward,
          path: [...env.pathTrail],
        });
      } else if (stepRes.info.collision) {
        setStatus('Collision');
        setIsRunning(false);
        setDqnResult({
          algorithm: 'DQN',
          success: false,
          pathLength: env.pathTrail.length - 1,
          steps: newStep,
          collisionCount: 1,
          totalReward: accumulatedReward + stepRes.reward,
          path: [...env.pathTrail],
        });
      } else {
        setStatus('Navigating');
      }
    }
  };

  // Continuous Play / Pause loop
  useEffect(() => {
    if (isRunning) {
      timerRef.current = setInterval(() => {
        handleStep();
      }, 250);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRunning, robotPos, status, algorithm, accumulatedReward, stepCount]);

  // Reset Robot & Simulation completely to start state
  const handleReset = (sPos: Position = startPos, tPos: Position = targetPos, clearDynamic: boolean = true) => {
    setIsRunning(false);
    if (timerRef.current) clearInterval(timerRef.current);
    if (clearDynamic) {
      env.clearDynamicObstacles();
      setDynamicObstacles(new Set());
      setObstacleAlert(null);
    }
    env.reset(sPos, tPos, clearDynamic);
    setRobotPos({ ...sPos });
    setPathTrail([{ ...sPos }]);
    setCurrentAction(null);
    setStepReward(0);
    setAccumulatedReward(0);
    setStepCount(0);
    setStatus('Ready');
    setDqnResult(null);
    setAstarResult(null);
    setQValues(agent.getQValues(env.getStateVector()));

    const currentBlocked = new Set<string>();
    shelves.forEach(k => currentBlocked.add(k));
    obstacles.forEach(k => currentBlocked.add(k));
    if (!clearDynamic) {
      dynamicObstacles.forEach(k => currentBlocked.add(k));
    }

    if (algorithm === 'A*') {
      const res = runAStar(env.width, env.height, sPos, tPos, currentBlocked);
      setPlannedPath(res.path);
    } else {
      const simEnv = new WarehouseEnvironment(env.width, env.height);
      simEnv.shelves = new Set(shelves);
      simEnv.obstacles = new Set(obstacles);
      simEnv.dynamicObstacles = new Set(dynamicObstacles);
      simEnv.startPos = { ...sPos };
      simEnv.target = { ...tPos };
      simEnv.robot = { ...sPos };

      const projected: Position[] = [{ ...sPos }];
      let simSteps = 0;
      while (simSteps < 25) {
        simSteps++;
        if (simEnv.robot.x === simEnv.target.x && simEnv.robot.y === simEnv.target.y) break;
        const dec = agent.selectBestNavigationAction(simEnv, projected, true);
        if (dec.action === 'STOP') break;
        const r = simEnv.step(dec.actionIndex);
        projected.push({ ...simEnv.robot });
        if (r.info.targetReached || r.info.collision) break;
      }
      setPlannedPath(projected.length > 1 ? projected : []);
    }
  };

  // Reset to original warehouse layout (Reset Map / Default Map)
  const handleResetMap = () => {
    if (isRunning) setIsRunning(false);
    if (timerRef.current) clearInterval(timerRef.current);
    env.loadMap(selectedMap);
    const newStart = { ...env.startPos };
    const newTarget = { ...env.target };
    setStartPos(newStart);
    setTargetPos(newTarget);
    setRobotPos(newStart);
    setShelves(new Set(env.shelves));
    setObstacles(new Set(env.obstacles));
    setDynamicObstacles(new Set());
    setObstacleAlert(null);
    setFeedbackMessage({ text: 'Warehouse layout restored to original state.', type: 'info' });
    agent.trainOnEnvironment(env);
    handleReset(newStart, newTarget, true);
  };

  // Clear Path or reset environment
  const handleClear = () => {
    handleReset(startPos, targetPos);
    env.clearTrail();
    setPathTrail([{ ...startPos }]);
  };

  // Warehouse Map Selection (5 predefined maps)
  const handleMapChange = (mapId: WarehouseMapId) => {
    if (isRunning) setIsRunning(false);
    if (timerRef.current) clearInterval(timerRef.current);
    setSelectedMap(mapId);
    env.loadMap(mapId);
    const newStart = { ...env.startPos };
    const newTarget = { ...env.target };
    setStartPos(newStart);
    setTargetPos(newTarget);
    setRobotPos(newStart);
    setShelves(new Set(env.shelves));
    setObstacles(new Set(env.obstacles));
    setDynamicObstacles(new Set());
    setObstacleAlert(null);
    setFeedbackMessage(null);

    // Train agent on the selected warehouse map layout
    agent.trainOnEnvironment(env);
    handleReset(newStart, newTarget, true);
    setNavExplanation(`Analyzed path for ${WAREHOUSE_MAPS.find(m => m.id === mapId)?.name || 'warehouse'}. Policy ready.`);
  };

  // Reset to Default Warehouse Map & Positions
  const handleResetToDefault = () => {
    handleMapChange('basic');
  };

  // Dynamic Obstacle (Special Innovation Feature): Drop obstacle 1 cell ahead in robot's current path
  const handleAddDynamicObstacleInPath = () => {
    const currentDir = currentAction && currentAction !== 'STOP' ? currentAction : 'RIGHT';
    const frontCell = env.getFrontCellInRobotPath(currentDir);
    if (!frontCell) return;

    // Get previous planned action from DQN before dynamic obstacle appears
    const prevDecision = agent.act(env.getStateVector(), true);
    const prevAct = ACTIONS[prevDecision.actionIndex];

    // Insert dynamic obstacle into warehouse environment
    env.addDynamicObstacle(frontCell.x, frontCell.y);
    setDynamicObstacles(new Set(env.dynamicObstacles));

    let blockedIdx = prevDecision.actionIndex;
    if (frontCell.x > env.robot.x) blockedIdx = 3;
    else if (frontCell.x < env.robot.x) blockedIdx = 2;
    else if (frontCell.y > env.robot.y) blockedIdx = 1;
    else if (frontCell.y < env.robot.y) blockedIdx = 0;

    // Senses obstacle -> Stops immediately -> Learns detour -> Immediately starts moving to destination
    triggerSuddenObstacleReaction(frontCell, prevAct, blockedIdx);
  };

  // Clear all dynamic obstacles
  const handleClearDynamicObstacles = () => {
    env.clearDynamicObstacles();
    setDynamicObstacles(new Set());
    setObstacleAlert(null);
    agent.trainOnEnvironment(env, 100);
    setQValues(agent.getQValues(env.getStateVector()));
  };

  // Interactive Grid Editing: Set Robot, Goal, Shelf, Obstacle, Dynamic Obstacle, or Eraser
  const handleCellClick = (x: number, y: number) => {
    // Dynamic Obstacles can be placed or removed anytime (even while robot is moving or paused)
    if (activeTool === 'DYNAMIC_OBSTACLE') {
      if ((x === robotPos.x && y === robotPos.y) || (x === targetPos.x && y === targetPos.y)) {
        setFeedbackMessage({ text: 'Cannot place dynamic obstacle on the robot or goal.', type: 'error' });
        return;
      }

      // Record previous planned action
      const prevDecision = agent.act(env.getStateVector(), true);
      const prevAct = ACTIONS[prevDecision.actionIndex];

      const isAdded = env.toggleDynamicObstacle(x, y);
      setDynamicObstacles(new Set(env.dynamicObstacles));

      if (isAdded) {
        setFeedbackMessage({ text: `Dynamic obstacle added at (${x}, ${y})`, type: 'info' });
        const dx = x - env.robot.x;
        const dy = y - env.robot.y;
        let blockedIdx = prevDecision.actionIndex;
        if (dx > 0) blockedIdx = 3;
        else if (dx < 0) blockedIdx = 2;
        else if (dy > 0) blockedIdx = 1;
        else if (dy < 0) blockedIdx = 0;

        triggerSuddenObstacleReaction({ x, y }, prevAct, blockedIdx);
      } else {
        setFeedbackMessage({ text: `Dynamic obstacle removed from (${x}, ${y})`, type: 'info' });
        setObstacleAlert(null);
        agent.trainOnEnvironment(env, 100);
        setQValues(agent.getQValues(env.getStateVector()));
      }
      return;
    }

    if (isRunning) return;

    if (activeTool === 'ROBOT') {
      // Reposition robot start with strict boundary & obstacle validation
      if (x < 0 || x >= env.width || y < 0 || y >= env.height) {
        setFeedbackMessage({ text: 'Invalid robot position: Outside warehouse bounds.', type: 'error' });
        return;
      }
      if (x === targetPos.x && y === targetPos.y) {
        setFeedbackMessage({ text: 'Invalid robot position: Cannot place robot on the goal cell.', type: 'error' });
        return;
      }
      const key = `${x},${y}`;
      if (shelves.has(key) || obstacles.has(key) || dynamicObstacles.has(key)) {
        setFeedbackMessage({ text: 'Invalid robot position: Cannot place robot on an obstacle or shelf.', type: 'error' });
        return;
      }

      const ok = env.setRobotStart(x, y);
      if (!ok) {
        setFeedbackMessage({ text: 'Invalid robot position', type: 'error' });
        return;
      }
      const newStart = { x, y };
      setStartPos(newStart);
      setRobotPos(newStart);
      setFeedbackMessage({ text: `Robot starting position updated to (${x}, ${y})`, type: 'success' });
      agent.trainOnEnvironment(env);
      handleReset(newStart, targetPos);
    } else if (activeTool === 'TARGET') {
      // Reposition destination goal with strict validation
      if (x < 0 || x >= env.width || y < 0 || y >= env.height) {
        setFeedbackMessage({ text: 'Invalid goal position: Outside warehouse bounds.', type: 'error' });
        return;
      }
      if (x === robotPos.x && y === robotPos.y) {
        setFeedbackMessage({ text: 'Invalid goal position: Cannot place goal on the robot cell.', type: 'error' });
        return;
      }
      const key = `${x},${y}`;
      if (shelves.has(key) || obstacles.has(key) || dynamicObstacles.has(key)) {
        setFeedbackMessage({ text: 'Invalid goal position: Cannot place goal on an obstacle or shelf.', type: 'error' });
        return;
      }

      const ok = env.setTarget(x, y);
      if (!ok) {
        setFeedbackMessage({ text: 'Invalid goal position', type: 'error' });
        return;
      }
      const newTarget = { x, y };
      setTargetPos(newTarget);
      setFeedbackMessage({ text: `Goal destination updated to (${x}, ${y})`, type: 'success' });
      agent.trainOnEnvironment(env);
      handleReset(startPos, newTarget);
    } else if (activeTool === 'OBSTACLE') {
      // Add or remove hazard obstacle
      if ((x === robotPos.x && y === robotPos.y) || (x === targetPos.x && y === targetPos.y)) {
        setFeedbackMessage({ text: 'Cannot place obstacle on the robot or goal.', type: 'error' });
        return;
      }
      const key = `${x},${y}`;
      if (obstacles.has(key) || shelves.has(key)) {
        env.obstacles.delete(key);
        env.shelves.delete(key);
        setObstacles(new Set(env.obstacles));
        setShelves(new Set(env.shelves));
        setFeedbackMessage({ text: `Obstacle removed from (${x}, ${y})`, type: 'info' });
      } else {
        env.obstacles.add(key);
        setObstacles(new Set(env.obstacles));
        setFeedbackMessage({ text: `Obstacle added at (${x}, ${y})`, type: 'success' });
      }
      agent.trainOnEnvironment(env);
      handleReset(startPos, targetPos);
    } else if (activeTool === 'SHELF') {
      // Add or remove shelf rack
      if ((x === startPos.x && y === startPos.y) || (x === targetPos.x && y === targetPos.y)) {
        setFeedbackMessage({ text: 'Cannot place shelf on the robot or goal.', type: 'error' });
        return;
      }
      env.setShelf(x, y);
      setShelves(new Set(env.shelves));
      setObstacles(new Set(env.obstacles));
      agent.trainOnEnvironment(env);
      handleReset(startPos, targetPos);
    } else if (activeTool === 'ERASER') {
      // Clear cell
      env.clearCell(x, y);
      setShelves(new Set(env.shelves));
      setObstacles(new Set(env.obstacles));
      setDynamicObstacles(new Set(env.dynamicObstacles));
      agent.trainOnEnvironment(env);
      handleReset(startPos, targetPos);
    }
  };

  // Train DQN (Runs 50 more episodes with progress)
  const handleTrainDQN = () => {
    setIsRunning(false);
    setIsTraining(true);
    setShowTrainingPanel(true);

    setTimeout(() => {
      agent.seedDemonstrations(env, 2);
      const newHistory = [...trainingHistory];
      for (let i = 0; i < 50; i++) {
        const metric = agent.trainSingleEpisode(env, true);
        newHistory.push(metric);
      }
      setTrainingHistory(newHistory);
      setIsTrained(true);
      setIsTraining(false);
      handleReset();
    }, 100);
  };

  // Retrain DQN from scratch (100 episodes)
  const handleRetrainFromScratch = () => {
    setIsRunning(false);
    setIsTraining(true);
    agent.reset();

    setTimeout(() => {
      agent.seedDemonstrations(env, 3);
      const newHistory: EpisodeMetrics[] = [];
      for (let i = 0; i < 100; i++) {
        const metric = agent.trainSingleEpisode(env, true);
        newHistory.push(metric);
      }
      setTrainingHistory(newHistory);
      setIsTrained(true);
      setIsTraining(false);
      handleReset();
    }, 100);
  };

  const latestEpisode = trainingHistory[trainingHistory.length - 1] || {
    episode: 0,
    totalReward: 0,
    steps: 0,
    success: false,
    epsilon: agent.epsilon,
  };

  // Find max Q-value key
  const maxQAction = useMemo(() => {
    let maxK = 'RIGHT';
    let maxV = -Infinity;
    (Object.keys(qValues) as (keyof QValues)[]).forEach(k => {
      if (qValues[k] > maxV) {
        maxV = qValues[k];
        maxK = k;
      }
    });
    return maxK;
  }, [qValues]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-sky-500 selection:text-slate-950 pb-16">
      {/* 1. HEADER */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-30 px-4 py-3 sm:py-4 shadow-lg">
        <div className="max-w-5xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono font-bold text-sky-400 uppercase tracking-widest bg-sky-950/70 border border-sky-800/80 px-2.5 py-0.5 rounded-full">
                SIMATS ENGINEERING • MLA0304
              </span>
              <span className="text-xs text-slate-500 hidden sm:inline">•</span>
              <span className="text-xs text-slate-400 hidden sm:inline">
                Authors: K. L. Khaviya Varshini &amp; M. Nandhini
              </span>
            </div>
            <h1 className="text-xl sm:text-2xl font-black text-slate-100 tracking-tight mt-1">
              Adaptive Path Planning for Autonomous Warehouse Robots
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
              2D Warehouse Navigation using Deep Reinforcement Learning (DQN)
            </p>
          </div>

          <div className="flex items-center gap-2 self-start md:self-auto">
            <div className="text-right text-[11px] font-mono text-slate-400 hidden sm:block">
              <span className="text-slate-200 font-bold block">Supervisor: Dr. Dumpala Prasanth</span>
              <span>Associate Professor, Dept. of CS &amp; AI</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-3 sm:px-6 py-6 space-y-6">
        {/* 2. WAREHOUSE SIMULATION */}
        <section className="bg-slate-900/90 rounded-3xl p-4 sm:p-6 border border-slate-800 shadow-xl space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Compass className="w-5 h-5 text-sky-400" />
              <h2 className="text-base sm:text-lg font-bold text-slate-100">
                2D Warehouse Simulation
              </h2>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <span className="font-mono text-slate-400">
                Current Algorithm:{' '}
                <strong className={algorithm === 'DQN' ? 'text-sky-400' : 'text-emerald-400'}>
                  {algorithm === 'DQN' ? 'DQN (Deep Q-Network)' : 'A* (Heuristic Search)'}
                </strong>
              </span>
            </div>
          </div>

          {/* Interactive Map Selection & Positions Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-950/80 p-3.5 rounded-2xl border border-slate-800">
            {/* Map Selection Dropdown */}
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-slate-300">Map:</span>
              <select
                value={selectedMap}
                onChange={(e) => handleMapChange(e.target.value as WarehouseMapId)}
                className="bg-slate-900 border border-slate-700 text-slate-100 rounded-lg px-3 py-1.5 text-xs font-semibold focus:outline-none focus:ring-2 focus:ring-sky-500 cursor-pointer"
                title="Select a predefined warehouse map"
              >
                {WAREHOUSE_MAPS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Positions & Customization Toggle */}
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 font-mono">
                <span className="w-2 h-2 rounded-full bg-emerald-400 ring-2 ring-emerald-950" />
                <span className="text-slate-400">Robot Start:</span>
                <span className="font-bold text-emerald-400 font-mono">({startPos.x}, {startPos.y})</span>
              </div>

              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 font-mono">
                <span className="w-2 h-2 rounded-full bg-rose-400 ring-2 ring-rose-950" />
                <span className="text-slate-400">Destination:</span>
                <span className="font-bold text-rose-400 font-mono">({targetPos.x}, {targetPos.y})</span>
              </div>

              {/* Toggle Customize Toolbar Button */}
              <button
                type="button"
                onClick={() => setShowCustomizeToolbar(!showCustomizeToolbar)}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5 border ${
                  showCustomizeToolbar
                    ? 'bg-sky-500 text-slate-950 border-sky-400 shadow-md shadow-sky-500/20'
                    : 'bg-slate-900 hover:bg-slate-800 text-slate-300 border-slate-750'
                }`}
                title="Customize Robot, Goal, Shelves, or Obstacles"
              >
                <SlidersHorizontal className="w-3.5 h-3.5" />
                <span>Customize Map</span>
              </button>

              {/* Reset to Default Button */}
              <button
                type="button"
                onClick={handleResetToDefault}
                className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent hover:border-slate-750 transition-all cursor-pointer flex items-center gap-1"
                title="Reset warehouse to Default Basic Map"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Reset to Default</span>
              </button>
            </div>
          </div>

          {/* Optional Grid Customization Toolbar (Revealed when clicked) */}
          {showCustomizeToolbar && (
            <div className="bg-slate-950/90 p-3 rounded-2xl border border-slate-800/90 flex flex-col gap-2 animate-in fade-in duration-150">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 text-xs text-slate-300 font-semibold">
                  <Wrench className="w-3.5 h-3.5 text-sky-400" />
                  <span>Placement Tool (Click cell to edit):</span>
                </div>
                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => {
                      setActiveTool('ROBOT');
                      setFeedbackMessage({ text: 'Click an empty cell to set the Robot starting position.', type: 'info' });
                    }}
                    className={`px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer flex items-center gap-1 border ${
                      activeTool === 'ROBOT'
                        ? 'bg-emerald-500 text-slate-950 border-emerald-400 shadow'
                        : 'bg-slate-900 text-slate-300 border-slate-800 hover:bg-slate-850'
                    }`}
                  >
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    <span>Select Robot</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setActiveTool('TARGET');
                      setFeedbackMessage({ text: 'Click an empty cell to set the Goal destination.', type: 'info' });
                    }}
                    className={`px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer flex items-center gap-1 border ${
                      activeTool === 'TARGET'
                        ? 'bg-rose-600 text-white border-rose-400 shadow'
                        : 'bg-slate-900 text-slate-300 border-slate-800 hover:bg-slate-850'
                    }`}
                  >
                    <span className="w-2 h-2 rounded-full bg-rose-400" />
                    <span>Select Goal</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setActiveTool('OBSTACLE');
                      setFeedbackMessage({ text: 'Click cells to add or remove obstacles.', type: 'info' });
                    }}
                    className={`px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer flex items-center gap-1 border ${
                      activeTool === 'OBSTACLE'
                        ? 'bg-rose-700 text-white border-rose-500 shadow'
                        : 'bg-slate-900 text-slate-300 border-slate-800 hover:bg-slate-850'
                    }`}
                  >
                    <span className="text-[10px]">✕</span>
                    <span>Add/Remove Obstacles</span>
                  </button>
                  <button
                    type="button"
                    onClick={handleResetMap}
                    className="px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer flex items-center gap-1 border bg-slate-900 hover:bg-slate-850 text-slate-300 border-slate-800 hover:border-slate-700"
                    title="Restore original warehouse layout"
                  >
                    <RotateCcw className="w-3 h-3" />
                    <span>Reset Map</span>
                  </button>
                </div>
              </div>
              {feedbackMessage && (
                <div className={`text-xs px-3 py-1.5 rounded-lg flex items-center justify-between gap-2 border animate-in fade-in duration-150 ${
                  feedbackMessage.type === 'error'
                    ? 'bg-rose-950/80 border-rose-500/80 text-rose-200'
                    : feedbackMessage.type === 'success'
                    ? 'bg-emerald-950/80 border-emerald-500/80 text-emerald-200'
                    : 'bg-slate-900 border-slate-750 text-slate-300'
                }`}>
                  <div className="flex items-center gap-2">
                    <span>{feedbackMessage.type === 'error' ? '⚠️' : feedbackMessage.type === 'success' ? '✅' : 'ℹ️'}</span>
                    <span className="font-medium">{feedbackMessage.text}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setFeedbackMessage(null)}
                    className="text-slate-400 hover:text-slate-200 text-xs px-1 cursor-pointer"
                  >
                    ✕
                  </button>
                </div>
              )}
              <div className="text-[11px] text-slate-400 bg-slate-900/60 px-3 py-1.5 rounded-lg border border-slate-850 flex items-center gap-2">
                <MousePointerClick className="w-3 h-3 text-sky-400 shrink-0" />
                <span>
                  Click any cell on the grid to change the position or toggle obstacles. The environment and DQN will update automatically.
                </span>
              </div>
            </div>
          )}

          {/* Dynamic Obstacle Insertion (Innovation Feature Bar) */}
          <div className="bg-gradient-to-r from-amber-950/40 via-slate-950 to-amber-950/30 p-3.5 rounded-2xl border border-amber-500/30 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-xl bg-amber-500/20 border border-amber-500/40 flex items-center justify-center text-amber-300 text-sm font-bold shadow-sm">
                ⚠️
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-bold text-amber-200">Dynamic Obstacle</span>
                  <span className="text-[9px] uppercase font-mono font-black bg-amber-500/20 text-amber-300 px-1.5 py-0.2 rounded border border-amber-500/30">
                    Innovation Feature
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">
                  Inject new obstacles during navigation to demonstrate real-time DQN state perception &amp; re-routing.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              {/* Drop in Robot's Path */}
              <button
                type="button"
                onClick={handleAddDynamicObstacleInPath}
                className="px-3 py-1.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs shadow-md shadow-amber-500/20 transition-all cursor-pointer flex items-center gap-1.5"
                title="Spawns an obstacle directly in the robot's upcoming path"
              >
                <Zap className="w-3.5 h-3.5 fill-current" />
                <span>Drop in Robot&apos;s Path</span>
              </button>

              {/* Click Grid to Add/Remove */}
              <button
                type="button"
                onClick={() => setActiveTool('DYNAMIC_OBSTACLE')}
                className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5 border ${
                  activeTool === 'DYNAMIC_OBSTACLE'
                    ? 'bg-amber-400/20 text-amber-200 border-amber-400 ring-2 ring-amber-400/30 shadow'
                    : 'bg-slate-900 hover:bg-slate-850 text-slate-300 border-slate-750'
                }`}
                title="Click any cell on the grid to toggle dynamic obstacles"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Add / Remove (Click Grid)</span>
              </button>

              {/* Clear Dynamic Obstacles */}
              {dynamicObstacles.size > 0 && (
                <button
                  type="button"
                  onClick={handleClearDynamicObstacles}
                  className="px-2.5 py-1.5 rounded-xl bg-slate-900 hover:bg-rose-950/50 text-slate-400 hover:text-rose-300 border border-slate-800 hover:border-rose-800/50 text-xs font-medium transition-all cursor-pointer flex items-center gap-1"
                  title="Remove all dynamic obstacles"
                >
                  <Trash2 className="w-3 h-3" />
                  <span>Clear Dynamic ({dynamicObstacles.size})</span>
                </button>
              )}
            </div>
          </div>

          {/* Dynamic Obstacle Notification Alert */}
          {obstacleAlert?.show && (
            <div className={`p-3.5 rounded-2xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 shadow-lg border transition-all duration-200 ${
              obstacleAlert.phase === 'STOPPED'
                ? 'bg-rose-950/60 border-rose-500/70 shadow-rose-950/40 text-rose-100'
                : obstacleAlert.phase === 'LEARNING'
                ? 'bg-amber-950/60 border-amber-500/70 shadow-amber-950/40 text-amber-100'
                : 'bg-emerald-950/60 border-emerald-500/70 shadow-emerald-950/40 text-emerald-100'
            }`}>
              <div className="flex items-center gap-3">
                <span className="text-xl">
                  {obstacleAlert.phase === 'STOPPED' ? '🛑' : obstacleAlert.phase === 'LEARNING' ? '🧠' : '🚀'}
                </span>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold font-mono">
                      {obstacleAlert.phase === 'STOPPED'
                        ? `Sudden Obstacle Sensed at (${obstacleAlert.pos.x}, ${obstacleAlert.pos.y}) — Robot STOPPED`
                        : obstacleAlert.phase === 'LEARNING'
                        ? `Observing Changed Environment — DQN Learning Detour...`
                        : `Detour Learned! Immediately Moving toward Destination`}
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-xs mt-0.5">
                    <span className="text-slate-300">
                      Previous Action: <strong className="text-rose-400 font-mono">{obstacleAlert.prevAction}</strong> → <span className="text-rose-300">Blocked</span>
                    </span>
                    <span className="text-slate-500">•</span>
                    <span className="text-slate-300">
                      New Action: <strong className={obstacleAlert.newAction === 'STOP' ? 'text-amber-400 font-mono' : 'text-emerald-400 font-mono'}>
                        {obstacleAlert.newAction}
                      </strong> → <span className={obstacleAlert.newAction === 'STOP' ? 'text-amber-300' : 'text-emerald-300 font-semibold'}>
                        {obstacleAlert.newAction === 'STOP' ? 'Halt & Evaluate' : 'Selected by DQN'}
                      </span>
                    </span>
                  </div>
                </div>
              </div>

              <div className="text-[11px] px-3 py-1 rounded-xl font-mono border bg-slate-950/60 border-slate-800 shrink-0">
                {obstacleAlert.phase === 'STOPPED' && 'Phase 1: Sensor Triggered Stop'}
                {obstacleAlert.phase === 'LEARNING' && 'Phase 2: Experience Replay & Q-Update'}
                {obstacleAlert.phase === 'RESUMING' && 'Phase 3: Resumed Movement'}
                {!obstacleAlert.phase && 'Real-Time DQN Adaptation'}
              </div>
            </div>
          )}

          {/* Centered 2D Grid Canvas */}
          <div className="flex justify-center py-1">
            <WarehouseGrid
              width={env.width}
              height={env.height}
              robot={robotPos}
              target={targetPos}
              startPos={startPos}
              shelves={shelves}
              obstacles={obstacles}
              dynamicObstacles={dynamicObstacles}
              pathTrail={pathTrail}
              plannedPath={plannedPath}
              onCellClick={handleCellClick}
              activeTool={activeTool}
              robotStatus={
                status === 'Goal Reached'
                  ? 'HAPPY'
                  : status === 'Collision'
                  ? 'COLLISION'
                  : obstacleAlert?.phase === 'STOPPED' || status === 'Alert / Stopped'
                  ? 'ALERT'
                  : obstacleAlert?.phase === 'LEARNING'
                  ? 'THINKING'
                  : isRunning
                  ? 'MOVING'
                  : 'IDLE'
              }
              lastAction={currentAction || 'RIGHT'}
              cellSize={46}
              algorithm={algorithm}
            />
          </div>
        </section>

        {/* 3. CONTROLS */}
        <section className="bg-slate-900/90 rounded-2xl p-4 border border-slate-800 shadow-md">
          <div className="flex flex-wrap items-center justify-between gap-3">
            {/* Algorithm Selector: DQN | A* */}
            <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
              <button
                onClick={() => {
                  setAlgorithm('DQN');
                  handleReset();
                }}
                className={`px-4 py-2 rounded-lg text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5 ${
                  algorithm === 'DQN'
                    ? 'bg-sky-500 text-slate-950 shadow-md shadow-sky-500/20'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Cpu className="w-3.5 h-3.5" />
                <span>DQN</span>
              </button>
              <button
                onClick={() => {
                  setAlgorithm('A*');
                  handleReset();
                }}
                className={`px-4 py-2 rounded-lg text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5 ${
                  algorithm === 'A*'
                    ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Compass className="w-3.5 h-3.5" />
                <span>A*</span>
              </button>
            </div>

            {/* Execution Buttons: Start | Step | Reset | Train DQN | Clear */}
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={() => {
                  if (status === 'Goal Reached') {
                    handleReset();
                  }
                  setIsRunning(!isRunning);
                }}
                className={`px-4 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer flex items-center gap-2 ${
                  isRunning
                    ? 'bg-amber-500 hover:bg-amber-400 text-slate-950'
                    : 'bg-sky-500 hover:bg-sky-400 text-slate-950 shadow-lg shadow-sky-500/20'
                }`}
                title={status === 'Goal Reached' ? 'Start a fresh run' : isRunning ? 'Pause simulation' : 'Start simulation'}
              >
                {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
                <span>{isRunning ? 'Pause' : 'Start'}</span>
              </button>

              <button
                onClick={handleStep}
                disabled={isRunning || status === 'Goal Reached'}
                className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold text-xs border border-slate-700 transition-all cursor-pointer flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
                title="Execute 1 action step"
              >
                <SkipForward className="w-3.5 h-3.5" />
                <span>Step</span>
              </button>

              <button
                onClick={() => handleReset()}
                className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold text-xs border border-slate-700 transition-all cursor-pointer flex items-center gap-1.5"
                title="Reset simulation to starting state"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset</span>
              </button>

              <button
                onClick={handleTrainDQN}
                disabled={isTraining}
                className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs border border-indigo-500/50 shadow-md shadow-indigo-500/20 transition-all cursor-pointer flex items-center gap-2 disabled:opacity-50"
              >
                <Flame className={`w-3.5 h-3.5 text-amber-300 ${isTraining ? 'animate-bounce' : ''}`} />
                <span>{isTraining ? 'Training...' : 'Train DQN (+50)'}</span>
              </button>

              <button
                onClick={handleClear}
                className="px-3 py-2 rounded-xl text-slate-400 hover:text-slate-200 text-xs font-semibold hover:bg-slate-800 transition-all cursor-pointer"
              >
                Clear Path
              </button>
            </div>
          </div>
        </section>

        {/* 4. CURRENT DQN STATUS */}
        <section className="bg-slate-900/90 rounded-2xl p-4 sm:p-5 border border-slate-800 shadow-lg space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono font-bold text-sky-400 uppercase tracking-wider flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5" />
              <span>Current {algorithm} Status &amp; Telemetry</span>
            </h3>
            <span
              className={`text-xs font-bold px-2.5 py-0.5 rounded-full ${
                status === 'Goal Reached'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                  : status === 'Collision'
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                  : status === 'Navigating'
                  ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40 animate-pulse'
                  : 'bg-slate-800 text-slate-400'
              }`}
            >
              Status: {status}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs">
            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 text-[10px] uppercase font-mono block">State</span>
              <span className="font-bold text-slate-100 mt-0.5 block truncate">
                Robot ({robotPos.x},{robotPos.y}), Goal ({env.target.x},{env.target.y})
              </span>
            </div>

            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 text-[10px] uppercase font-mono block">Selected Action</span>
              <span className="font-bold text-sky-400 mt-0.5 block text-sm font-mono">
                {currentAction || '—'}
              </span>
            </div>

            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 text-[10px] uppercase font-mono block">Step Reward</span>
              <span
                className={`font-bold mt-0.5 block text-sm font-mono ${
                  stepReward > 0 ? 'text-emerald-400' : stepReward < 0 ? 'text-rose-400' : 'text-slate-300'
                }`}
              >
                {stepReward >= 0 ? `+${stepReward}` : stepReward}
              </span>
            </div>

            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 text-[10px] uppercase font-mono block">Training Episode</span>
              <span className="font-bold text-slate-200 mt-0.5 block text-sm font-mono">
                #{agent.episodeCount}
              </span>
            </div>

            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 col-span-2 sm:col-span-1">
              <span className="text-slate-400 text-[10px] uppercase font-mono block">Steps</span>
              <span className="font-bold text-slate-200 mt-0.5 block text-sm font-mono">
                {stepCount}
              </span>
            </div>
          </div>

          {/* Live Q-Values Output: Q(s, a) */}
          {algorithm === 'DQN' && (
            <div className="pt-2 border-t border-slate-800/80">
              <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1.5 font-mono">
                <span>DQN Output Q(s, a) for Available Actions:</span>
                <span className="text-[10px] text-slate-500">Max Q-Value Action Chosen (Argmax)</span>
              </div>
              <div className="grid grid-cols-5 gap-2 text-center text-xs">
                {(['UP', 'DOWN', 'LEFT', 'RIGHT', 'STOP'] as (keyof QValues)[]).map(act => {
                  const isMax = maxQAction === act;
                  return (
                    <div
                      key={act}
                      className={`p-1.5 rounded-lg border font-mono transition-all ${
                        isMax
                          ? 'bg-sky-500/20 border-sky-400 text-sky-300 font-bold shadow-sm shadow-sky-500/30'
                          : 'bg-slate-950/60 border-slate-800 text-slate-400'
                      }`}
                    >
                      <div className="text-[10px]">{act}</div>
                      <div className="text-xs font-bold mt-0.5">{qValues[act]}</div>
                    </div>
                  );
                })}
              </div>

              {/* Path Analysis & Real-Time Reasoning Box */}
              <div className="mt-3 p-3 rounded-xl bg-slate-950/80 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
                <div className="flex items-start sm:items-center gap-2.5">
                  <div
                    className={`w-7 h-7 rounded-lg flex items-center justify-center text-xs font-mono font-bold shrink-0 ${
                      loopDetectedNotice
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                        : 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                    }`}
                  >
                    {loopDetectedNotice ? '↻' : '🎯'}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] uppercase font-mono font-bold text-slate-400">
                        Path Analysis &amp; Goal Guidance
                      </span>
                      {loopDetectedNotice && (
                        <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          Loop Avoidance Detour
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-300 font-mono mt-0.5">
                      {navExplanation}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-xs font-mono shrink-0 text-slate-400">
                  <div className="bg-slate-900/90 px-2.5 py-1 rounded-lg border border-slate-800">
                    Manhattan to Goal: <span className="text-sky-300 font-bold">{Math.abs(robotPos.x - env.target.x) + Math.abs(robotPos.y - env.target.y)}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* 5. SIMPLE PERFORMANCE (DQN vs A*) */}
        <section className="bg-slate-900/90 rounded-2xl p-4 sm:p-5 border border-slate-800 shadow-md space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono font-bold text-sky-400 uppercase tracking-wider flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5" />
              <span>Simple Performance Comparison (Actual Calculated Values)</span>
            </h3>
            <span className="text-[11px] text-slate-500">Benchmark on current 10x10 layout</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px] font-mono">
                  <th className="py-2 px-3">Metric</th>
                  <th className="py-2 px-3 text-sky-400">DQN (Deep Q-Network)</th>
                  <th className="py-2 px-3 text-emerald-400">A* (Heuristic Search)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-200">Goal Reached</td>
                  <td className="py-2.5 px-3">
                    {dqnResult ? (
                      dqnResult.success ? (
                        <span className="text-emerald-400 font-bold flex items-center gap-1">
                          <CheckCircle2 className="w-3.5 h-3.5" /> Yes
                        </span>
                      ) : (
                        <span className="text-rose-400 font-bold">No</span>
                      )
                    ) : (
                      <span className="text-slate-500">Run DQN to evaluate</span>
                    )}
                  </td>
                  <td className="py-2.5 px-3">
                    {astarResult ? (
                      astarResult.success ? (
                        <span className="text-emerald-400 font-bold flex items-center gap-1">
                          <CheckCircle2 className="w-3.5 h-3.5" /> Yes
                        </span>
                      ) : (
                        <span className="text-rose-400 font-bold">No</span>
                      )
                    ) : (
                      <span className="text-slate-500">Run A* to evaluate</span>
                    )}
                  </td>
                </tr>

                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-200">Path Length</td>
                  <td className="py-2.5 px-3">
                    {dqnResult ? `${dqnResult.pathLength} cells` : '—'}
                  </td>
                  <td className="py-2.5 px-3">
                    {astarResult ? `${astarResult.pathLength} cells` : '—'}
                  </td>
                </tr>

                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-200">Steps Taken</td>
                  <td className="py-2.5 px-3">
                    {dqnResult ? `${dqnResult.steps} steps` : '—'}
                  </td>
                  <td className="py-2.5 px-3">
                    {astarResult ? `${astarResult.steps} steps` : '—'}
                  </td>
                </tr>

                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-200">Collisions</td>
                  <td className="py-2.5 px-3">
                    {dqnResult ? (
                      <span className={dqnResult.collisionCount === 0 ? 'text-emerald-400' : 'text-rose-400'}>
                        {dqnResult.collisionCount}
                      </span>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td className="py-2.5 px-3">
                    {astarResult ? (
                      <span className={astarResult.collisionCount === 0 ? 'text-emerald-400' : 'text-rose-400'}>
                        {astarResult.collisionCount}
                      </span>
                    ) : (
                      '—'
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* 6. SIMPLE TRAINING SECTION */}
        <section className="bg-slate-900/90 rounded-2xl p-4 sm:p-5 border border-slate-800 shadow-md space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <h3 className="text-xs font-mono font-bold text-sky-400 uppercase tracking-wider flex items-center gap-1.5">
                <Flame className="w-3.5 h-3.5 text-amber-400" />
                <span>Simple Training Panel (DQN Learning Telemetry)</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Monitors real episode rewards, steps, and exploration decay (&epsilon;).
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleTrainDQN}
                disabled={isTraining}
                className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-all cursor-pointer disabled:opacity-50"
              >
                {isTraining ? 'Training...' : 'Train 50 More Episodes'}
              </button>
              <button
                onClick={handleRetrainFromScratch}
                disabled={isTraining}
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition-all cursor-pointer disabled:opacity-50"
              >
                Reset &amp; Retrain
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Total Episodes</span>
              <span className="text-base font-bold font-mono text-slate-100 mt-0.5 block">
                {latestEpisode.episode}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Latest Episode Reward</span>
              <span
                className={`text-base font-bold font-mono mt-0.5 block ${
                  latestEpisode.totalReward > 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {latestEpisode.totalReward}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Latest Steps</span>
              <span className="text-base font-bold font-mono text-slate-100 mt-0.5 block">
                {latestEpisode.steps}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Exploration Rate (&epsilon;)</span>
              <span className="text-base font-bold font-mono text-sky-400 mt-0.5 block">
                {(latestEpisode.epsilon * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Interactive Recharts Line Chart: Total Reward per Episode */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between flex-wrap gap-2 text-xs">
              <span className="font-mono flex items-center gap-1.5 text-slate-200 font-bold">
                <TrendingUp className="w-4 h-4 text-sky-400" />
                Total Reward per Episode (DQN Learning Curve)
              </span>
              <div className="flex items-center gap-3 text-[11px] font-mono text-slate-400">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-sky-400 inline-block" />
                  <span className="text-slate-300">Total Reward</span>
                </span>
                <span className="text-slate-600">•</span>
                <span className="text-slate-400">{trainingHistory.length} Episodes Tracked</span>
              </div>
            </div>

            {trainingHistory.length > 0 ? (
              <div className="h-60 w-full pt-1">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={trainingHistory}
                    margin={{ top: 10, right: 20, left: 0, bottom: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="episode"
                      stroke="#475569"
                      tick={{ fontSize: 10, fill: '#94a3b8' }}
                      tickLine={false}
                      label={{
                        value: 'Episode',
                        position: 'insideBottomRight',
                        offset: -5,
                        fill: '#64748b',
                        fontSize: 10,
                      }}
                    />
                    <YAxis
                      stroke="#475569"
                      tick={{ fontSize: 10, fill: '#94a3b8' }}
                      tickLine={false}
                      label={{
                        value: 'Total Reward',
                        angle: -90,
                        position: 'insideLeft',
                        offset: 15,
                        fill: '#64748b',
                        fontSize: 10,
                      }}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const d = payload[0].payload as EpisodeMetrics;
                          return (
                            <div className="bg-slate-900 border border-slate-700 p-2.5 rounded-xl shadow-xl text-xs font-mono space-y-1 z-50">
                              <div className="text-slate-200 font-bold border-b border-slate-800 pb-1">
                                Episode #{d.episode}
                              </div>
                              <div className="text-sky-400 font-bold">
                                Total Reward: {d.totalReward}
                              </div>
                              <div className="text-slate-300">Steps: {d.steps}</div>
                              <div className="text-amber-400">
                                Exploration (&epsilon;): {(d.epsilon * 100).toFixed(1)}%
                              </div>
                              <div className={d.success ? 'text-emerald-400 font-semibold' : 'text-rose-400'}>
                                Result: {d.success ? 'Goal Reached (+100)' : 'Collision / Max Steps'}
                              </div>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <ReferenceLine y={0} stroke="#334155" strokeDasharray="3 3" />
                    <Line
                      type="monotone"
                      dataKey="totalReward"
                      name="Total Reward"
                      stroke="#38bdf8"
                      strokeWidth={2}
                      dot={trainingHistory.length <= 40 ? { r: 2.5, fill: '#38bdf8' } : false}
                      activeDot={{ r: 5, fill: '#38bdf8', stroke: '#0284c7', strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-40 flex flex-col items-center justify-center text-center p-4 border border-dashed border-slate-800 rounded-xl bg-slate-900/40 space-y-2">
                <TrendingUp className="w-6 h-6 text-slate-600" />
                <p className="text-xs text-slate-400 max-w-sm">
                  No training history recorded yet. Click{' '}
                  <strong className="text-sky-400">Train 50 Episodes</strong> or{' '}
                  <strong className="text-sky-400">Train 100 Episodes</strong> above to see the interactive convergence graph.
                </p>
              </div>
            )}

            <div className="flex justify-between text-[10px] font-mono text-slate-500 pt-0.5">
              <span>← Initial Random Exploration</span>
              <span>Reward Convergence (Learned Collision-Free Path) →</span>
            </div>
          </div>
        </section>

        {/* 7. HOW IT WORKS */}
        <section className="bg-slate-900/90 rounded-2xl p-5 sm:p-6 border border-slate-800 shadow-lg space-y-5">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <GraduationCap className="w-5 h-5 text-sky-400" />
            <h2 className="text-base sm:text-lg font-bold text-slate-100">
              HOW IT WORKS
            </h2>
          </div>

          {/* Module 1 & Module 2 Definitions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <div className="flex items-center gap-2 text-sky-400 font-bold text-xs font-mono">
                <span className="w-5 h-5 rounded-full bg-sky-500/20 flex items-center justify-center text-[10px]">
                  1
                </span>
                <span>MODULE 1: Warehouse Environment &amp; State Management</span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                “Creates the warehouse environment and provides the robot&apos;s current state, destination and nearby obstacles.”
              </p>
              <div className="text-[11px] text-slate-400 space-y-1 pt-1">
                <div>• Represents 2D grid with Robot, Goal, Shelves, and Static Obstacles.</div>
                <div>• Encodes 10-dimensional state: position, goal relative vector, and 4-way sensors.</div>
                <div>• Detects boundaries, static collisions, and destination reached.</div>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs font-mono">
                <span className="w-5 h-5 rounded-full bg-emerald-500/20 flex items-center justify-center text-[10px]">
                  2
                </span>
                <span>MODULE 2: DQN-Based Adaptive Path Planning</span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                “Uses DQN to learn which movement is best for reaching the destination while avoiding obstacles.”
              </p>
              <div className="text-[11px] text-slate-400 space-y-1 pt-1">
                <div>• Q-Network maps input state to 5 discrete actions: UP, DOWN, LEFT, RIGHT, STOP.</div>
                <div>• Uses Experience Replay buffer to eliminate sample correlation.</div>
                <div>• Target Network synchronizes periodically to stabilize Bellman target updates.</div>
              </div>
            </div>
          </div>

          {/* Innovation – Dynamic Obstacle Avoidance */}
          <div className="p-4 rounded-xl bg-gradient-to-br from-amber-950/30 to-slate-950 border border-amber-500/40 space-y-2.5">
            <div className="flex items-center gap-2 text-amber-300 font-bold text-xs font-mono">
              <span className="w-5 h-5 rounded-full bg-amber-500/20 flex items-center justify-center text-[10px]">
                ★
              </span>
              <span>INNOVATION: Dynamic Obstacle Avoidance</span>
            </div>
            <p className="text-xs text-slate-200 leading-relaxed italic bg-amber-950/40 p-3 rounded-lg border border-amber-500/30">
              “During navigation, new obstacles can be introduced into the warehouse. The DQN observes the changed environment and selects a new action to avoid the obstacle and continue toward the destination.”
            </p>
            <div className="pt-1">
              <span className="text-[11px] font-mono text-slate-400 block mb-1.5 font-semibold">
                Dynamic Perception &amp; Reaction Pipeline:
              </span>
              <div className="flex flex-wrap items-center gap-1.5 text-[10px] font-mono text-slate-300">
                <span className="px-2 py-1 rounded bg-slate-900 border border-amber-500/40 text-amber-300 font-bold">
                  New obstacle appears
                </span>
                <span className="text-slate-500">→</span>
                <span className="px-2 py-1 rounded bg-slate-900 border border-slate-800">
                  Environment changes
                </span>
                <span className="text-slate-500">→</span>
                <span className="px-2 py-1 rounded bg-slate-900 border border-sky-500/40 text-sky-300">
                  State updates (Sensors)
                </span>
                <span className="text-slate-500">→</span>
                <span className="px-2 py-1 rounded bg-slate-900 border border-emerald-500/40 text-emerald-300 font-bold">
                  DQN selects new action
                </span>
                <span className="text-slate-500">→</span>
                <span className="px-2 py-1 rounded bg-slate-900 border border-slate-800">
                  Robot avoids obstacle
                </span>
                <span className="text-slate-500">→</span>
                <span className="px-2 py-1 rounded bg-slate-900 border border-rose-500/40 text-rose-300">
                  Continues toward destination
                </span>
              </div>
            </div>
          </div>

          {/* 9-Step Flow Sequence */}
          <div className="space-y-2 pt-2">
            <span className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider block">
              9-Step Decision &amp; Learning Cycle
            </span>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
              {[
                { step: '1', title: 'Create Warehouse', desc: 'Initialize 2D grid, start pos, shelves, & goal.' },
                { step: '2', title: 'Observe Robot & Obstacles', desc: 'Sense immediate surroundings and corridor bounds.' },
                { step: '3', title: 'Create Current State', desc: 'Vectorize coordinates and 4-direction sensor status.' },
                { step: '4', title: 'DQN Selects Action', desc: 'Compute Q(s, a) and choose optimal movement.' },
                { step: '5', title: 'Robot Moves', desc: 'Execute action cell-by-cell in the warehouse.' },
                { step: '6', title: 'Environment Gives Reward', desc: '+100 goal, -30 collision, step & distance feedback.' },
                { step: '7', title: 'DQN Learns', desc: 'Mini-batch training via Experience Replay & Bellman loss.' },
                { step: '8', title: 'Robot Improves Path', desc: 'Q-values converge, eliminating oscillations and loops.' },
                { step: '9', title: 'Robot Reaches Destination', desc: 'AMR arrives at goal via collision-free trajectory.' },
              ].map(item => (
                <div key={item.step} className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-1">
                  <div className="flex items-center gap-1.5 font-bold text-slate-200">
                    <span className="text-[10px] font-mono text-sky-400 bg-sky-950 px-1.5 py-0.5 rounded border border-sky-800/60">
                      Step {item.step}
                    </span>
                    <span className="text-xs">{item.title}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-snug">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-4 px-4 text-xs text-slate-500 text-center">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>
            Adaptive Path Planning for Autonomous Warehouse Robots • SIMATS ENGINEERING
          </span>
          <span className="text-slate-400">
            K. L. Khaviya Varshini (192472381) &amp; M. Nandhini (192472318)
          </span>
        </div>
      </footer>
    </div>
  );
}
