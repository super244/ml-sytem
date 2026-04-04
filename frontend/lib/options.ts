import type { Difficulty, OutputFormat, SolverMode } from '@/lib/api';

export const DIFFICULTY_OPTIONS: Array<{ value: Difficulty; label: string }> = [
  { value: 'easy', label: 'Easy' },
  { value: 'medium', label: 'Medium' },
  { value: 'hard', label: 'Hard' },
  { value: 'olympiad', label: 'Olympiad' },
];

export const SOLVER_MODE_OPTIONS: Array<{ value: SolverMode; label: string }> = [
  { value: 'rigorous', label: 'Rigorous' },
  { value: 'exam', label: 'Exam' },
  { value: 'concise', label: 'Concise' },
  { value: 'verification', label: 'Verification' },
];

export const OUTPUT_FORMAT_OPTIONS: Array<{ value: OutputFormat; label: string }> = [
  { value: 'text', label: 'Text' },
  { value: 'json', label: 'Structured' },
];

export const SAMPLE_OPTIONS = [1, 3, 5];
