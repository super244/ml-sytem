import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import Layout from '@/components/factory/Layout';
import PageHeader from '@/components/factory/PageHeader';
import GlassCard from '@/components/factory/GlassCard';
import { apiRequest } from '@/lib/api';
import { Send, Flag, ChevronDown, Loader2 } from 'lucide-react';

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
  const [isSending, setIsSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const send = async () => {
    if (!input.trim() || isSending) return;
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsSending(true);

    try {
      const response = await apiRequest<any>('/inference/completions', {
        method: 'POST',
        body: JSON.stringify({
          model,
          messages: [...messages, userMsg].map(m => ({ role: m.role, content: m.content })),
        }),
      });

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response?.choices?.[0]?.message?.content || response?.content || response?.text || String(response),
        confidence: response?.confidence,
        tokens: response?.usage?.total_tokens || response?.tokens,
        tps: response?.tps,
        ttft: response?.ttft,
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch {
      const fallbackMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Unable to reach the inference server. The API may not be running. This is a fallback response.',
        confidence: 0,
        tokens: 0,
        tps: 0,
        ttft: 0,
      };
      setMessages(prev => [...prev, fallbackMsg]);
    } finally {
      setIsSending(false);
    }
  };

  const latestAssistant = [...messages].reverse().find(m => m.role === 'assistant');

  return (
    <Layout>
      <motion.div variants={pageVariants} initial="initial" animate="animate" className="h-screen flex flex-col">
        <PageHeader title="Inference" />
        <div className="flex-1 flex min-h-0">
          <div className="flex-1 flex flex-col min-w-0">
            <div className="px-4 py-2 border-b border-border flex items-center gap-2">
              <span className="text-xs font-mono text-muted-foreground">Model:</span>
              <button data-testid="button-model-selector" className="text-xs font-mono text-foreground bg-raised px-3 py-1 rounded-lg border border-border flex items-center gap-1.5 hover:bg-overlay transition-colors">
                {model} <ChevronDown className="w-3 h-3" />
              </button>
            </div>

            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4" data-testid="chat-messages">
              {messages.map(msg => (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`} data-testid={`message-${msg.role}-${msg.id}`}>
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
                    {msg.role === 'assistant' && msg.confidence !== undefined && (
                      <div className="flex items-center gap-3 mt-3 pt-2 border-t border-border">
                        <span className="text-[10px] font-mono text-muted-foreground">
                          conf: <span className="text-neon-green">{msg.confidence}</span>
                        </span>
                        <button data-testid={`button-flag-${msg.id}`} className="text-[10px] font-mono text-neon-red/60 hover:text-neon-red transition-colors flex items-center gap-1">
                          <Flag className="w-3 h-3" /> Flag
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isSending && (
                <div className="flex justify-start">
                  <div className="glass-panel rounded-2xl rounded-bl-sm px-4 py-3">
                    <Loader2 className="w-4 h-4 animate-spin text-neon-blue" />
                  </div>
                </div>
              )}
            </div>

            <div className="p-4 border-t border-border">
              <div className="flex items-center gap-2">
                <input
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && send()}
                  placeholder="Send a message..."
                  disabled={isSending}
                  data-testid="input-chat-message"
                  className="flex-1 bg-raised border border-border rounded-xl px-4 py-2.5 text-sm font-display text-foreground placeholder:text-muted-foreground outline-none focus:ring-1 focus:ring-neon-blue/30 disabled:opacity-50"
                />
                <button
                  onClick={send}
                  disabled={isSending}
                  data-testid="button-send-message"
                  className="p-2.5 rounded-xl bg-neon-blue/20 border border-neon-blue/30 text-neon-blue hover:bg-neon-blue/30 transition-colors disabled:opacity-50"
                >
                  {isSending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>

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
              <button data-testid="button-export-chat" className="w-full text-[10px] font-mono px-2 py-1.5 rounded-lg bg-raised border border-border text-muted-foreground hover:text-foreground transition-colors">
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
    <div className="text-lg font-mono text-foreground" data-testid={`text-telemetry-${label.toLowerCase()}`}>{value}<span className="text-xs text-muted-foreground ml-1">{unit}</span></div>
  </div>
);

export default InferenceChat;
