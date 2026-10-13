import React from 'react';
import { Action } from '../types';

interface CuteRobotProps {
  status?: 'IDLE' | 'MOVING' | 'THINKING' | 'HAPPY' | 'ALERT' | 'COLLISION';
  lastAction?: Action;
  size?: number;
  className?: string;
}

export const CuteRobot: React.FC<CuteRobotProps> = ({
  status = 'IDLE',
  lastAction = 'RIGHT',
  size = 40,
  className = '',
}) => {
  // Determine rotation based on action
  let rotation = 0;
  if (lastAction === 'UP') rotation = -90;
  else if (lastAction === 'DOWN') rotation = 90;
  else if (lastAction === 'LEFT') rotation = 180;
  else if (lastAction === 'RIGHT') rotation = 0;

  const isAlert = status === 'ALERT' || status === 'COLLISION';
  const isHappy = status === 'HAPPY';
  const isThinking = status === 'THINKING';

  return (
    <div
      className={`relative inline-flex items-center justify-center select-none transition-transform duration-200 ${className}`}
      style={{ width: size, height: size }}
      title={`AMR Robot [Status: ${status}, Direction: ${lastAction}]`}
    >
      <svg
        viewBox="0 0 100 100"
        className="w-full h-full drop-shadow-md transition-all duration-300"
        style={{
          transform: `rotate(${rotation}deg)`,
        }}
      >
        <defs>
          <linearGradient id="robotBodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#38BDF8" />
            <stop offset="50%" stopColor="#0284C7" />
            <stop offset="100%" stopColor="#0369A1" />
          </linearGradient>

          <linearGradient id="chassisGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#334155" />
            <stop offset="100%" stopColor="#0F172A" />
          </linearGradient>

          <filter id="robotGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Shadow */}
        <ellipse cx="50" cy="88" rx="36" ry="7" fill="rgba(0,0,0,0.25)" />

        {/* Heavy-duty Tread / Wheels */}
        <rect x="14" y="20" width="12" height="60" rx="6" fill="#1E293B" stroke="#475569" strokeWidth="2" />
        <rect x="74" y="20" width="12" height="60" rx="6" fill="#1E293B" stroke="#475569" strokeWidth="2" />

        {/* Wheel treads markings */}
        <line x1="14" y1="35" x2="26" y2="35" stroke="#64748B" strokeWidth="2" />
        <line x1="14" y1="50" x2="26" y2="50" stroke="#64748B" strokeWidth="2" />
        <line x1="14" y1="65" x2="26" y2="65" stroke="#64748B" strokeWidth="2" />

        <line x1="74" y1="35" x2="86" y2="35" stroke="#64748B" strokeWidth="2" />
        <line x1="74" y1="50" x2="86" y2="50" stroke="#64748B" strokeWidth="2" />
        <line x1="74" y1="65" x2="86" y2="65" stroke="#64748B" strokeWidth="2" />

        {/* Main Robot Rounded Chassis */}
        <rect
          x="22"
          y="18"
          width="56"
          height="64"
          rx="18"
          fill="url(#robotBodyGrad)"
          stroke="#BAE6FD"
          strokeWidth="2"
        />

        {/* Front Direction Indicator (Bumper/LiDAR bar) */}
        <path
          d="M 80 44 L 88 50 L 80 56 Z"
          fill="#38BDF8"
          className={status === 'MOVING' ? 'animate-pulse' : ''}
        />

        {/* Top Scanner / Antenna Puck */}
        <circle cx="36" cy="30" r="5" fill="#0284C7" stroke="#38BDF8" strokeWidth="1.5" />
        <circle
          cx="36"
          cy="30"
          r="2.5"
          fill={isAlert ? '#EF4444' : isThinking ? '#F59E0B' : '#10B981'}
          className={isThinking ? 'animate-ping' : ''}
        />

        {/* Digital Visor / Screen Face */}
        <rect
          x="30"
          y="38"
          width="40"
          height="24"
          rx="8"
          fill="#0F172A"
          stroke="#38BDF8"
          strokeWidth="1.5"
        />

        {/* Robot Eyes (Expressive Screen) */}
        {isHappy ? (
          // Happy curved eyes ^ ^
          <g stroke="#34D399" strokeWidth="3" strokeLinecap="round" fill="none">
            <path d="M 38 52 Q 42 44 46 52" />
            <path d="M 54 52 Q 58 44 62 52" />
          </g>
        ) : isAlert ? (
          // Alert X X or ! !
          <g stroke="#F87171" strokeWidth="2.5" strokeLinecap="round">
            <line x1="39" y1="46" x2="45" y2="54" />
            <line x1="45" y1="46" x2="39" y2="54" />
            <line x1="55" y1="46" x2="61" y2="54" />
            <line x1="61" y1="46" x2="55" y2="54" />
          </g>
        ) : isThinking ? (
          // Thinking dots / processing eyes
          <g fill="#FBBF24">
            <circle cx="41" cy="50" r="3.5" className="animate-pulse" />
            <circle cx="59" cy="50" r="3.5" className="animate-pulse" />
          </g>
        ) : (
          // Normal Friendly glowing LED eyes
          <g fill="#38BDF8">
            <ellipse cx="41" cy="50" rx="3.5" ry="4.5" />
            <ellipse cx="59" cy="50" rx="3.5" ry="4.5" />
            <circle cx="42" cy="48" r="1.2" fill="#FFFFFF" />
            <circle cx="60" cy="48" r="1.2" fill="#FFFFFF" />
          </g>
        )}

        {/* Cute blush cheeks */}
        {!isAlert && (
          <g fill="#F472B6" opacity="0.6">
            <circle cx="34" cy="56" r="2" />
            <circle cx="66" cy="56" r="2" />
          </g>
        )}

        {/* Battery & Status Strip on chassis rear */}
        <rect x="34" y="68" width="32" height="4" rx="2" fill="#0369A1" />
        <rect
          x="36"
          y="69"
          width="20"
          height="2"
          rx="1"
          fill={isAlert ? '#EF4444' : '#10B981'}
        />
      </svg>
    </div>
  );
};
