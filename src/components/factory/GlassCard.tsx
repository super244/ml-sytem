import { ReactNode } from 'react';

interface GlassCardProps {
  children: ReactNode;
  glow?: 'green' | 'blue' | 'orange' | 'red' | 'purple' | 'none';
  hover?: boolean;
  className?: string;
  onClick?: () => void;
}

const GlassCard = ({ children, glow = 'none', hover = false, className = '', onClick }: GlassCardProps) => {
  const glowClass = glow !== 'none' ? `glow-${glow}` : '';
  const hoverClass = hover ? 'glass-card' : 'glass-panel';

  return (
    <div
      className={`${hoverClass} ${glowClass} p-4 ${className}`}
      onClick={onClick}
    >
      {children}
    </div>
  );
};

export default GlassCard;
