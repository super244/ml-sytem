import GlassCard from './GlassCard';

export const LoadingSkeleton = ({ rows = 3 }: { rows?: number }) => (
  <div className="space-y-4" data-testid="loading-skeleton">
    {Array.from({ length: rows }, (_, i) => (
      <GlassCard key={i}>
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-raised rounded w-1/3" />
          <div className="h-3 bg-raised rounded w-2/3" />
          <div className="h-3 bg-raised rounded w-1/2" />
        </div>
      </GlassCard>
    ))}
  </div>
);

export const ErrorState = ({ message, onRetry }: { message: string; onRetry?: () => void }) => (
  <GlassCard glow="red" data-testid="error-state">
    <div className="text-center py-8">
      <div className="text-sm font-mono text-neon-red mb-2">Error loading data</div>
      <div className="text-xs font-mono text-muted-foreground mb-4">{message}</div>
      {onRetry && (
        <button
          onClick={onRetry}
          data-testid="button-retry"
          className="text-xs font-mono px-4 py-2 rounded-lg bg-raised border border-border text-muted-foreground hover:text-foreground transition-colors"
        >
          Retry
        </button>
      )}
    </div>
  </GlassCard>
);
