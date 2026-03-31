import { useEffect, useState } from 'react';

interface CountUpProps {
  end: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  className?: string;
}

const CountUp = ({ end, duration = 600, prefix = '', suffix = '', decimals = 0, className = '' }: CountUpProps) => {
  const [value, setValue] = useState(0);

  useEffect(() => {
    const startTime = Date.now();
    const tick = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(eased * end);
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [end, duration]);

  return (
    <span className={`metric-value ${className}`}>
      {prefix}{value.toFixed(decimals)}{suffix}
    </span>
  );
};

export default CountUp;
