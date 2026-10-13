import React from 'react';
import { motion } from 'motion/react';
import { Action, EditTool, Position } from '../types';
import { CuteRobot } from './CuteRobot';

interface WarehouseGridProps {
  width: number;
  height: number;
  robot: Position;
  target: Position;
  startPos: Position;
  shelves: Set<string>;
  obstacles: Set<string>;
  dynamicObstacles?: Set<string>;
  pathTrail?: Position[];
  plannedPath?: Position[];
  onCellClick?: (x: number, y: number) => void;
  robotStatus?: 'IDLE' | 'MOVING' | 'THINKING' | 'HAPPY' | 'ALERT' | 'COLLISION';
  lastAction?: Action;
  cellSize?: number;
  activeTool?: EditTool;
  algorithm?: 'DQN' | 'A*';
}

export const WarehouseGrid: React.FC<WarehouseGridProps> = ({
  width,
  height,
  robot,
  target,
  startPos,
  shelves,
  obstacles,
  dynamicObstacles = new Set<string>(),
  pathTrail = [],
  plannedPath = [],
  onCellClick,
  robotStatus = 'IDLE',
  lastAction = 'RIGHT',
  cellSize = 48,
  activeTool,
  algorithm = 'DQN',
}) => {
  const gapPx = 4;
  const padPx = 12;

  // Path trail lookup
  const pathSet = React.useMemo(() => {
    const map = new Map<string, number>();
    pathTrail.forEach((p, idx) => {
      map.set(`${p.x},${p.y}`, idx + 1);
    });
    return map;
  }, [pathTrail]);

  // Planned path lookup
  const plannedSet = React.useMemo(() => {
    const set = new Set<string>();
    plannedPath.forEach(p => set.add(`${p.x},${p.y}`));
    return set;
  }, [plannedPath]);

  return (
    <div className="flex flex-col items-center select-none w-full">
      {/* Grid Canvas Frame */}
      <div className="p-3 sm:p-4 bg-slate-900 rounded-2xl border border-slate-800 shadow-2xl overflow-auto max-w-full">
        <div
          className="relative grid bg-slate-950 rounded-xl border border-slate-800/80"
          style={{
            gridTemplateColumns: `repeat(${width}, ${cellSize}px)`,
            gridTemplateRows: `repeat(${height}, ${cellSize}px)`,
            gap: `${gapPx}px`,
            padding: `${padPx}px`,
          }}
        >
          {Array.from({ length: height }).map((_, y) =>
            Array.from({ length: width }).map((_, x) => {
              const key = `${x},${y}`;
              const isRobot = robot.x === x && robot.y === y;
              const isTarget = target.x === x && target.y === y;
              const isStart = startPos.x === x && startPos.y === y;
              const isShelf = shelves.has(key);
              const isObstacle = obstacles.has(key);
              const isDynamicObstacle = dynamicObstacles.has(key);
              const trailStep = pathSet.get(key);
              const isPlanned = plannedSet.has(key) && !isRobot && !isTarget;

              let cellBg = 'bg-slate-900/60 border-slate-800/60';
              if (isDynamicObstacle) {
                cellBg = 'bg-amber-950/90 border-amber-500 text-amber-200 shadow-md shadow-amber-500/20';
              } else if (isShelf) {
                cellBg = 'bg-amber-950/70 border-amber-700/70 text-amber-300';
              } else if (isObstacle) {
                cellBg = 'bg-rose-950/70 border-rose-700/80 text-rose-300';
              } else if (trailStep && !isRobot && !isTarget) {
                cellBg = 'bg-sky-950/50 border-sky-500/40';
              } else if (isPlanned) {
                cellBg = 'bg-emerald-950/40 border-emerald-500/40';
              }

              let hoverBorder = 'hover:border-slate-500';
              if (activeTool === 'ROBOT') {
                hoverBorder = 'hover:border-emerald-400 hover:shadow-lg hover:shadow-emerald-500/20';
              } else if (activeTool === 'TARGET') {
                hoverBorder = 'hover:border-rose-400 hover:shadow-lg hover:shadow-rose-500/20';
              } else if (activeTool === 'SHELF') {
                hoverBorder = 'hover:border-amber-400 hover:shadow-lg hover:shadow-amber-500/20';
              } else if (activeTool === 'OBSTACLE') {
                hoverBorder = 'hover:border-rose-500 hover:shadow-lg hover:shadow-rose-500/20';
              } else if (activeTool === 'DYNAMIC_OBSTACLE') {
                hoverBorder = 'hover:border-amber-400 hover:ring-2 hover:ring-amber-500/50';
              } else if (activeTool === 'ERASER') {
                hoverBorder = 'hover:border-slate-400';
              }

              return (
                <div
                  key={key}
                  onClick={() => onCellClick?.(x, y)}
                  className={`relative rounded-lg flex items-center justify-center transition-all duration-150 cursor-pointer overflow-hidden border ${cellBg} ${hoverBorder}`}
                  style={{ width: cellSize, height: cellSize }}
                  title={`(${x}, ${y}) - ${
                    isRobot
                      ? 'Robot Start'
                      : isTarget
                      ? 'Goal Destination'
                      : isDynamicObstacle
                      ? 'Dynamic Obstacle (Appeared in path)'
                      : isShelf
                      ? 'Storage Shelf'
                      : isObstacle
                      ? 'Static Obstacle'
                      : 'Free Floor'
                  }`}
                >
                  {/* Subtle coordinate label */}
                  {!isRobot && !isTarget && !isShelf && !isObstacle && !isDynamicObstacle && (
                    <span className="absolute top-0.5 left-1 text-[8px] text-slate-400 font-mono select-none">
                      {x},{y}
                    </span>
                  )}

                  {/* Start Point Marker (when robot is elsewhere) */}
                  {isStart && !isRobot && (
                    <div className="absolute inset-1 rounded border border-dashed border-emerald-500/40 flex items-center justify-center bg-emerald-950/20">
                      <span className="text-[8px] font-bold text-emerald-400/70 font-mono">START</span>
                    </div>
                  )}

                  {/* Dynamic Obstacle (Special Innovation Feature) ⚠️ */}
                  {isDynamicObstacle && (
                    <div className="relative w-full h-full p-1 flex flex-col items-center justify-center bg-gradient-to-br from-amber-950 to-orange-950/90">
                      <div className="absolute inset-0 border-2 border-amber-400 rounded animate-pulse opacity-70" />
                      <div className="text-[11px] font-bold text-amber-300 animate-bounce">⚠️</div>
                      <span className="text-[7px] font-black tracking-tighter text-amber-200 font-mono uppercase">
                        DYNAMIC
                      </span>
                    </div>
                  )}

                  {/* Warehouse Storage Shelf ⬛ */}
                  {isShelf && !isDynamicObstacle && (
                    <div className="w-full h-full p-1 flex flex-col justify-between">
                      <div className="h-1 bg-amber-500/80 rounded-full" />
                      <div className="flex items-center justify-center text-[8px] font-mono font-bold text-amber-300">
                        SHELF
                      </div>
                      <div className="h-1 bg-amber-500/80 rounded-full" />
                    </div>
                  )}

                  {/* Static Obstacle (Crate/Hazard) ⬛ */}
                  {isObstacle && !isDynamicObstacle && (
                    <div className="w-full h-full p-1 flex items-center justify-center">
                      <div className="w-full h-full rounded bg-rose-900/60 border border-rose-600/60 flex items-center justify-center">
                        <span className="text-xs font-black text-rose-300">✕</span>
                      </div>
                    </div>
                  )}

                  {/* Destination / Goal 🔴 */}
                  {isTarget && (
                    <div className="relative w-full h-full flex flex-col items-center justify-center">
                      <div className="absolute w-7 h-7 rounded-full bg-rose-500/30 animate-ping" />
                      <div className="w-7 h-7 rounded-full bg-rose-600 border-2 border-rose-300 flex items-center justify-center shadow-lg shadow-rose-600/50 z-10">
                        <span className="text-xs">🎯</span>
                      </div>
                      <span className="text-[7px] font-black text-rose-300 font-mono uppercase tracking-tighter mt-0.5">
                        GOAL
                      </span>
                    </div>
                  )}

                  {/* Path Trail Breadcrumb */}
                  {trailStep && !isRobot && !isTarget && (
                    <div className="flex flex-col items-center justify-center">
                      <div className="w-2.5 h-2.5 rounded-full bg-sky-400 shadow-sm shadow-sky-400" />
                      <span className="text-[7px] text-sky-400 font-mono mt-0.5">{trailStep}</span>
                    </div>
                  )}

                  {/* Planned / Analyzed Path Ghost Dot */}
                  {isPlanned && !trailStep && !isRobot && !isTarget && (
                    <div className="relative flex items-center justify-center">
                      <div
                        className={`w-2.5 h-2.5 rounded-full ${
                          algorithm === 'DQN'
                            ? 'bg-sky-400 shadow-sm shadow-sky-400/50'
                            : 'bg-emerald-400 shadow-sm shadow-emerald-400/50'
                        }`}
                      />
                      <div
                        className={`absolute -inset-1 rounded-full animate-ping opacity-25 ${
                          algorithm === 'DQN' ? 'bg-sky-400' : 'bg-emerald-400'
                        }`}
                      />
                    </div>
                  )}

                  {/* Robot Floor Shadow Under AMR */}
                  {isRobot && (
                    <div className="absolute inset-2 rounded-full bg-emerald-500/10 border border-emerald-500/20" />
                  )}
                </div>
              );
            })
          )}

          {/* Animated Autonomous Mobile Robot (AMR) - smoothly glides across grid */}
          <motion.div
            initial={false}
            animate={{
              x: padPx + robot.x * (cellSize + gapPx),
              y: padPx + robot.y * (cellSize + gapPx),
            }}
            transition={{
              type: 'spring',
              damping: 24,
              stiffness: 240,
              mass: 0.7,
            }}
            className="absolute top-0 left-0 z-30 pointer-events-none flex items-center justify-center"
            style={{
              width: cellSize,
              height: cellSize,
            }}
          >
            <div className="relative flex items-center justify-center">
              <div
                className={`absolute inset-0 -m-1 rounded-full border-2 transition-colors duration-200 ${
                  robotStatus === 'COLLISION'
                    ? 'border-rose-400 bg-rose-500/30 ring-4 ring-rose-500/20'
                    : robotStatus === 'ALERT'
                    ? 'border-amber-400 bg-amber-500/30 ring-4 ring-amber-500/20 animate-pulse'
                    : robotStatus === 'HAPPY'
                    ? 'border-emerald-400 bg-emerald-500/30 ring-4 ring-emerald-500/20'
                    : 'border-emerald-400/80 bg-emerald-500/20 animate-pulse'
                }`}
              />
              <CuteRobot
                status={robotStatus}
                lastAction={lastAction}
                size={cellSize - 8}
              />
            </div>
          </motion.div>
        </div>
      </div>

      {/* Grid Legend */}
      <div className="mt-3 flex flex-wrap items-center justify-center gap-4 text-xs text-slate-400 bg-slate-900/70 px-4 py-2 rounded-xl border border-slate-800">
        <div className="flex items-center gap-1.5">
          <span className="text-sm">🟢</span>
          <span className="font-semibold text-slate-200">Robot (Start: {startPos.x},{startPos.y})</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-sm">🔴</span>
          <span className="font-semibold text-slate-200">Destination (Goal: {target.x},{target.y})</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3.5 h-3.5 rounded bg-amber-800 border border-amber-600 inline-block" />
          <span className="text-slate-300">Shelves</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3.5 h-3.5 rounded bg-rose-900 border border-rose-600 text-[8px] font-bold text-center inline-flex items-center justify-center text-rose-200">✕</span>
          <span className="text-slate-300">Obstacles</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3.5 h-3.5 rounded bg-amber-900 border border-amber-500 text-[8px] font-bold text-center inline-flex items-center justify-center text-amber-200 animate-pulse">⚠️</span>
          <span className="text-amber-300 font-medium">Dynamic Obstacle</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-sky-400 inline-block" />
          <span className="text-slate-300">Path Taken</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className={`w-2.5 h-2.5 rounded-full inline-block ${
              algorithm === 'DQN' ? 'bg-sky-400/80 shadow-sm shadow-sky-400/40' : 'bg-emerald-400/80 shadow-sm shadow-emerald-400/40'
            }`}
          />
          <span className="text-slate-300">
            {algorithm === 'DQN' ? 'DQN Analyzed Path' : 'A* Planned Path'}
          </span>
        </div>
        <div className="text-[11px] text-slate-500 italic hidden sm:inline">
          (Click any cell to toggle shelf/obstacle)
        </div>
      </div>
    </div>
  );
};
