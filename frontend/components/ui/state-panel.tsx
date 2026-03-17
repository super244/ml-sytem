import type { ReactNode } from "react";

type StatePanelProps = {
  eyebrow: string;
  title: string;
  description: string;
  tone?: "default" | "error" | "loading";
  action?: ReactNode;
};

export function StatePanel({
  eyebrow,
  title,
  description,
  tone = "default",
  action,
}: StatePanelProps) {
  return (
    <section className={`panel state-panel ${tone}`}>
      <div className="eyebrow">{eyebrow}</div>
      <h2 className="state-title">{title}</h2>
      <p className="hero-copy">{description}</p>
      {action ? <div className="state-action">{action}</div> : null}
    </section>
  );
}
