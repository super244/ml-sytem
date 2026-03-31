import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import Layout from '@/components/factory/Layout';
import PageHeader from '@/components/factory/PageHeader';
import GlassCard from '@/components/factory/GlassCard';
import { Send, Flag, ChevronDown } from 'lucide-react';

const pageVariants = {
  initial: { opacity: 0, y: 12, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.25 } },
};

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  confidence?: number;
  tokens?: number;
  tps?: number;
  ttft?: number;
}

const initialMessages: Message[] = [
  { id: '1', role: 'user', content: 'Explain the chain of thought reasoning approach for solving math problems.' },
  {
    id: '2', role: 'assistant',
    content: 'Chain of thought (CoT) reasoning breaks complex math problems into sequential logical steps. Instead of jumping directly to an answer, the model:\n\n1. **Identifies** the problem type and relevant formulas\n2. **Decomposes** the problem into sub-steps\n3. **Executes** each step while maintaining intermediate results\n4. **Verifies** the final answer against constraints\n\nThis approach improves accuracy on GSM8K benchmarks by ~15% compared to direct prompting, as it allows the model to leverage its parametric knowledge at each reasoning step.',
    confidence: 0.94, tokens: 847, tps: 42.3, ttft: 187,
  },
];

const InferenceChat = () => {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState('');
  const [model, setModel] = useState('llama-3.1-8b-math');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const send = () => {
    if (!input.trim()) return;
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');

    // Simulate response
    setTimeout(() => {
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'This is a simulated response from the model. In production, this would stream from the inference server via the /api/v1/inference/completions endpoint.',
        confidence: 0.87,
        tokens: 342,
        tps: 38.7,
        ttft: 203,
      };
      setMessages(prev => [...prev, assistantMsg]);
    }, 800);
  };

  const latestAssistant = [...messages].reverse().find(m => m.role === 'assistant');

  return (
    <Layout>
      <motion.div variants={pageVariants} initial="initial" animate="animate" className="h-screen flex flex-col">
        <PageHeader title="Inference" />
        <div className="flex-1 flex min-h-0">
          {/* Chat Panel */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Model Selector */}
            <div className="px-4 py-2 border-b border-border flex items-center gap-2">
              <span className="text-xs font-mono text-muted-foreground">Model:</span>
              <button className="text-xs font-mono text-foreground bg-raised px-3 py-1 rounded-lg border border-border flex items-center gap-1.5 hover:bg-overlay transition-colors">
                {model} <ChevronDown className="w-3 h-3" />
              </button>
            </div>

            {/* Messages */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map(msg => (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] ${msg.role === 'user'
                    ? 'bg-neon-blue/10 border border-neon-blue/20 rounded-2xl rounded-br-sm px-4 py-3'
                    : 'glass-panel rounded-2xl rounded-bl-sm px-4 py-3'
                  }`}>
                    <div className="text-[10px] font-mono text-muted-foreground mb-1.5 uppercase">
                      {msg.role}
                    </div>
                    <div className="text-sm font-display text-foreground/90 whitespace-pre-wrap leading-relaxed">
                      {msg.content}
                    </div>
                    {msg.role === 'assistant' && msg.confidence && (
                      <div className="flex items-center gap-3 mt-3 pt-2 border-t border-border">
                        <span className="text-[10px] font-mono text-muted-foreground">
                          conf: <span className="text-neon-green">{msg.confidence}</span>
                        </span>
                        <button className="text-[10px] font-mono text-neon-red/60 hover:text-neon-red transition-colors flex items-center gap-1">
                          <Flag className="w-3 h-3" /> Flag
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Input */}
            <div className="p-4 border-t border-border">
              <div className="flex items-center gap-2">
                <input
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && send()}
                  placeholder="Send a message..."
                  className="flex-1 bg-raised border border-border rounded-xl px-4 py-2.5 text-sm font-display text-foreground placeholder:text-muted-foreground outline-none focus:ring-1 focus:ring-neon-blue/30"
                />
                <button
                  onClick={send}
                  className="p-2.5 rounded-xl bg-neon-blue/20 border border-neon-blue/30 text-neon-blue hover:bg-neon-blue/30 transition-colors"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Telemetry Panel */}
          <div className="w-52 border-l border-border p-4 hidden lg:block">
            <div className="section-label mb-4">TELEMETRY</div>
            <div className="space-y-4">
              <TelemetryMetric label="TPS" value={latestAssistant?.tps ? `${latestAssistant.tps}` : '—'} unit="tok/s" />
              <TelemetryMetric label="TTFT" value={latestAssistant?.ttft ? `${latestAssistant.ttft}` : '—'} unit="ms" />
              <TelemetryMetric label="Tokens" value={latestAssistant?.tokens ? `${latestAssistant.tokens}` : '—'} unit="" />
              <div>
                <div className="text-[10px] font-mono text-muted-foreground mb-1">Confidence</div>
                <div className="h-2 bg-raised rounded-full overflow-hidden">
                  <div className="h-full bg-neon-green rounded-full progress-bar-transition" style={{ width: `${(latestAssistant?.confidence || 0) * 100}%` }} />
                </div>
                <div className="text-xs font-mono text-foreground mt-1">{latestAssistant?.confidence || '—'}</div>
              </div>
              <button className="w-full text-[10px] font-mono px-2 py-1.5 rounded-lg bg-raised border border-border text-muted-foreground hover:text-foreground transition-colors">
                ↗ Export Chat
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    </Layout>
  );
};

const TelemetryMetric = ({ label, value, unit }: { label: string; value: string; unit: string }) => (
  <div>
    <div className="text-[10px] font-mono text-muted-foreground">{label}</div>
    <div className="text-lg font-mono text-foreground">{value}<span className="text-xs text-muted-foreground ml-1">{unit}</span></div>
  </div>
);

export default InferenceChat;
