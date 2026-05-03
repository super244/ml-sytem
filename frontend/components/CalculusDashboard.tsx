'use client';
import React, { useState } from 'react';

// Assuming we have some MathJax or KaTeX wrapper available in the project.
// import 'katex/dist/katex.min.css';
// import { BlockMath } from 'react-katex';

export default function CalculusDashboard() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResponse('');
    try {
      const res = await fetch('/api/inference/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      setResponse(data.text);
    } catch (err) {
      setResponse('Error connecting to Inference API');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto bg-slate-900 text-white rounded-xl shadow-lg mt-10">
      <h1 className="text-3xl font-bold mb-4">Calculus 8B Inference Engine</h1>
      <form onSubmit={handleSubmit} className="mb-6">
        <textarea 
          className="w-full p-4 rounded-lg bg-slate-800 border border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500 mb-4"
          rows={4}
          placeholder="Enter a calculus problem (e.g., Compute the derivative of sin(x) * x^2)..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <button 
          type="submit" 
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded-lg font-semibold transition"
        >
          {loading ? 'Computing...' : 'Solve'}
        </button>
      </form>

      {response && (
        <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
          <h2 className="text-xl font-semibold mb-2 text-gray-300">Step-by-Step Solution:</h2>
          <div className="whitespace-pre-wrap font-mono text-sm leading-relaxed">
            {/* Ideally replace with <BlockMath math={response} /> for real LaTeX rendering */}
            {response}
          </div>
        </div>
      )}
    </div>
  );
}
