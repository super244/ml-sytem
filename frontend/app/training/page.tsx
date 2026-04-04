'use client';

import { useState } from 'react';
import Link from 'next/link';

const trainingMethods = [
  {
    id: 'pretraining',
    name: 'Pretraining',
    description: 'Train a model from scratch on large corpus',
    icon: '🚀',
    difficulty: 'Advanced',
    compute: 'High',
    duration: 'Weeks',
    profile: 'pretraining.yaml',
  },
  {
    id: 'full_finetune',
    name: 'Full Fine-tuning',
    description: 'Update all model parameters on task-specific data',
    icon: '⚡',
    difficulty: 'Intermediate',
    compute: 'Medium',
    duration: 'Days',
    profile: 'full_finetune.yaml',
  },
  {
    id: 'multitask_learning',
    name: 'Multitask Learning',
    description: 'Train on multiple tasks simultaneously',
    icon: '🎯',
    difficulty: 'Advanced',
    compute: 'High',
    duration: 'Weeks',
    profile: 'multitask_learning.yaml',
  },
  {
    id: 'continual_learning',
    name: 'Continual Learning',
    description: 'Learn continuously without forgetting previous tasks',
    icon: '🔄',
    difficulty: 'Advanced',
    compute: 'Medium',
    duration: 'Ongoing',
    profile: 'continual_learning.yaml',
  },
  {
    id: 'qlora',
    name: 'QLoRA Fine-tuning',
    description: 'Parameter-efficient fine-tuning with quantization',
    icon: '🔧',
    difficulty: 'Beginner',
    compute: 'Low',
    duration: 'Hours',
    profile: 'baseline_qlora.yaml',
  },
  {
    id: 'curriculum_learning',
    name: 'Curriculum Learning',
    description: 'Progressive training from easy to hard examples',
    icon: '📚',
    difficulty: 'Intermediate',
    compute: 'Medium',
    duration: 'Days',
    profile: 'curriculum_specialist.yaml',
  },
];

export default function TrainingPage() {
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null);

  const selectedMethodData = trainingMethods.find((m) => m.id === selectedMethod);

  return (
    <div className="training-container">
      <div className="training-header">
        <Link href="/dashboard" className="back-link">
          ← Back to Dashboard
        </Link>
        <h1>Training Methods</h1>
        <p>Choose the right training approach for your goals and resources</p>
      </div>

      <div className="training-content">
        <div className="methods-grid">
          {trainingMethods.map((method) => (
            <div
              key={method.id}
              className={`method-card ${selectedMethod === method.id ? 'selected' : ''}`}
              onClick={() => setSelectedMethod(method.id)}
            >
              <div className="method-icon">{method.icon}</div>
              <div className="method-content">
                <h3>{method.name}</h3>
                <p>{method.description}</p>
                <div className="method-meta">
                  <span className="meta-item">
                    <span className="meta-label">Difficulty:</span>
                    <span className={`meta-value difficulty-${method.difficulty.toLowerCase()}`}>
                      {method.difficulty}
                    </span>
                  </span>
                  <span className="meta-item">
                    <span className="meta-label">Compute:</span>
                    <span className={`meta-value compute-${method.compute.toLowerCase()}`}>
                      {method.compute}
                    </span>
                  </span>
                  <span className="meta-item">
                    <span className="meta-label">Duration:</span>
                    <span className="meta-value">{method.duration}</span>
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {selectedMethodData && (
          <div className="method-details">
            <h2>{selectedMethodData.name}</h2>
            <p className="method-description">{selectedMethodData.description}</p>

            <div className="details-grid">
              <div className="detail-card">
                <h3>Requirements</h3>
                <ul>
                  <li>
                    Dataset:{' '}
                    {selectedMethodData.id === 'pretraining'
                      ? 'Large corpus (100GB+)'
                      : 'Task-specific data'}
                  </li>
                  <li>
                    GPU:{' '}
                    {selectedMethodData.compute === 'High'
                      ? '8x A100 or equivalent'
                      : selectedMethodData.compute === 'Medium'
                        ? '2-4x A100'
                        : '1x RTX 4090'}
                  </li>
                  <li>
                    Memory:{' '}
                    {selectedMethodData.compute === 'High'
                      ? '320GB+'
                      : selectedMethodData.compute === 'Medium'
                        ? '80GB+'
                        : '24GB+'}
                  </li>
                  <li>Storage: {selectedMethodData.compute === 'High' ? '10TB+' : '500GB+'}</li>
                </ul>
              </div>

              <div className="detail-card">
                <h3>Configuration</h3>
                <ul>
                  <li>
                    Profile: <code>{selectedMethodData.profile}</code>
                  </li>
                  <li>
                    Learning Rate:{' '}
                    {selectedMethodData.id === 'pretraining'
                      ? '5e-5'
                      : selectedMethodData.id === 'full_finetune'
                        ? '1e-5'
                        : '1e-4'}
                  </li>
                  <li>
                    Batch Size:{' '}
                    {selectedMethodData.compute === 'High'
                      ? '1024'
                      : selectedMethodData.compute === 'Medium'
                        ? '256'
                        : '32'}
                  </li>
                  <li>
                    Epochs:{' '}
                    {selectedMethodData.id === 'continual_learning'
                      ? '50+'
                      : selectedMethodData.id === 'pretraining'
                        ? '10'
                        : '3-5'}
                  </li>
                </ul>
              </div>

              <div className="detail-card">
                <h3>Use Cases</h3>
                <ul>
                  {selectedMethodData.id === 'pretraining' && (
                    <>
                      <li>Building domain-specific models</li>
                      <li>Training on proprietary data</li>
                      <li>Creating base models for fine-tuning</li>
                    </>
                  )}
                  {selectedMethodData.id === 'full_finetune' && (
                    <>
                      <li>Domain adaptation</li>
                      <li>Style transfer</li>
                      <li>Complete model specialization</li>
                    </>
                  )}
                  {selectedMethodData.id === 'multitask_learning' && (
                    <>
                      <li>Cross-domain reasoning</li>
                      <li>Versatile assistant models</li>
                      <li>Resource-efficient training</li>
                    </>
                  )}
                  {selectedMethodData.id === 'continual_learning' && (
                    <>
                      <li>Always-learning systems</li>
                      <li>Catastrophic forgetting prevention</li>
                      <li>Adaptive AI assistants</li>
                    </>
                  )}
                  {selectedMethodData.id === 'qlora' && (
                    <>
                      <li>Quick prototyping</li>
                      <li>Resource-constrained training</li>
                      <li>Experimentation and research</li>
                    </>
                  )}
                  {selectedMethodData.id === 'curriculum_learning' && (
                    <>
                      <li>Complex skill acquisition</li>
                      <li>Improved convergence</li>
                      <li>Educational AI systems</li>
                    </>
                  )}
                </ul>
              </div>
            </div>

            <div className="action-buttons">
              <button className="primary-button">
                Start Training with {selectedMethodData.name}
              </button>
              <button className="secondary-button">View Configuration</button>
              <button className="secondary-button">Estimate Resources</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
