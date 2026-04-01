import { ReactNode, HTMLAttributes } from 'react';

interface GlassCardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  glow?: 'green' | 'blue' | 'orange' | 'red' | 'purple' | 'none';
  hover?: boolean;
}

const GlassCard = ({ children, glow = 'none', hover = false, className = '', ...rest }: GlassCardProps) => {
  const glowClass = glow !== 'none' ? `glow-${glow}` : '';
  const hoverClass = hover ? 'glass-card' : 'glass-panel';

  return (
    <div
      className={`${hoverClass} ${glowClass} p-4 ${className}`}
      {...rest}
    >
      {children}
    </div>
  );
};

export default GlassCard;
