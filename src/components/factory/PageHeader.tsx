import { ReactNode } from 'react';

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}

const PageHeader = ({ title, subtitle, actions }: PageHeaderProps) => (
  <div className="flex items-center justify-between px-6 py-4 border-b border-border">
    <div>
      <h1 className="text-xl font-display font-semibold text-foreground">{title}</h1>
      {subtitle && <p className="text-xs font-mono text-muted-foreground mt-0.5">{subtitle}</p>}
    </div>
    <div className="flex items-center gap-2">
      {actions}
      <kbd className="text-[10px] font-mono text-muted-foreground bg-raised px-2 py-1 rounded border border-border">
        ⌘K
      </kbd>
    </div>
  </div>
);

export default PageHeader;
