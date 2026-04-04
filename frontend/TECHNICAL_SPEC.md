# AI-Factory Frontend Technical Specification

## Overview

This document provides a comprehensive technical map of the AI-Factory frontend application, including all pages, API connections, state management patterns, and complex data flows. This serves as the "Logic Map" for redesign and architectural decisions.

## Technology Stack

- **Framework**: Next.js 16.2.1 with App Router
- **Language**: TypeScript 5.8.2
- **Styling**: Tailwind CSS 3.4.19
- **State Management**: React hooks (useState, useEffect, useTransition)
- **UI Components**: Custom component library with CSS-in-JS approach
- **Math Rendering**: KaTeX 0.16.11 with remark-math/rehype-katex
- **Validation**: Zod 3.25.76 for runtime type checking
- **Utilities**: clsx 2.1.1 for conditional styling

## Application Architecture

### Core Structure

```
frontend/
├── app/                    # Next.js App Router pages
├── components/             # Reusable UI components
├── lib/                    # Core utilities and API layer
├── hooks/                  # Custom React hooks
└── public/                 # Static assets
```

### Layout System

- **AppShell**: Main layout wrapper with navigation
- **AppNav**: Sticky navigation bar with active state management
- **PageHeader**: Consistent page headers with metrics and actions
- **StatePanel**: Error/loading/empty state components

## Pages and Routes

### Primary Pages

#### 1. Home Page (`/`)

- **File**: `app/page.tsx`
- **Purpose**: Redirects to `/dashboard`
- **State**: None
- **API Calls**: None

#### 2. Dashboard (`/dashboard`)

- **File**: `app/dashboard/page.tsx`
- **Purpose**: Mission control overview with lifecycle management
- **Key State Variables**:
  - `mission: MissionControlSnapshot | null` - Complete system state
  - `loading: boolean` - Loading state
  - `error: string | null` - Error state
  - `busyAction: string | null` - Action in progress
  - `notice: string | null` - User notifications
- **API Connections**:
  - `getMissionControl()` - Polls every 7.5 seconds
  - `createManagedInstance()` - Launch new instances
  - `runManagedInstanceAction()` - Trigger actions
- **Complex Features**:
  - Auto-refresh mission control data
  - Instance lifecycle management (train → finetune → evaluate → deploy)
  - Recommendation system integration
  - Titan runtime status display

#### 3. Solve Workspace (`/solve`)

- **File**: `app/solve/page.tsx` → `components/chat-shell.tsx`
- **Purpose**: Interactive math reasoning interface
- **Key State Variables**:
  - `messages: ChatMessage[]` - Chat history
  - `question: string` - Current input
  - `modelVariant: ModelVariant` - Selected model
  - `compareToModel: string` - Comparison model
  - `showReasoning: boolean` - Display reasoning
  - `difficultyTarget: Difficulty` - Problem difficulty
  - `useCalculator: boolean` - Calculator access
  - `solverMode: SolverMode` - Reasoning style
  - `temperature: number` - Generation temperature
  - `numSamples: number` - Candidate count
  - `promptPreset: string` - Prompt template
  - `outputFormat: OutputFormat` - Response format
  - `selectedCandidateIndex: number` - Candidate selection
  - `workspaceMode: WorkspaceMode` - UI density
  - `density: WorkspaceDensity` - Layout density
- **API Connections**:
  - `generateAnswer()` - Primary generation endpoint
  - `useLabMetadata()` - Models and prompts
- **Complex Features**:
  - Multi-candidate generation and ranking
  - Real-time candidate comparison
  - Mathematical expression rendering (KaTeX)
  - Persistent workspace preferences
  - Verifier-aware reasoning display

#### 4. Compare Lab (`/compare`)

- **File**: `app/compare/page.tsx` → `components/compare-lab.tsx`
- **Purpose**: Side-by-side model comparison
- **Key State Variables**:
  - `question: string` - Comparison prompt
  - `primaryModel: string` - Primary model
  - `secondaryModel: string` - Secondary model
  - `difficultyTarget: Difficulty` - Problem difficulty
  - `solverMode: SolverMode` - Reasoning style
  - `outputFormat: OutputFormat` - Response format
  - `promptPreset: string` - Prompt template
  - `result: CompareResponse | null` - Comparison results
  - `requestError: string | null` - Error state
- **API Connections**:
  - `compareModels()` - Direct comparison endpoint
  - `useLabMetadata()` - Models and prompts
- **Complex Features**:
  - Synchronized model comparison
  - Latency and quality metrics
  - Answer agreement analysis

#### 5. Runs Overview (`/runs`)

- **File**: `app/runs/page.tsx` → `components/runs-view.tsx`
- **Purpose**: Instance management and orchestration
- **Key State Variables**:
  - `instances: InstanceSummary[]` - All instances
  - `runs: OrchestrationRun[]` - Orchestration runs
  - `summary: OrchestrationSummary | null` - System summary
  - `loading: boolean` - Loading state
  - `error: string | null` - Error state
- **API Connections**:
  - `getInstances()` - Instance list
  - `getOrchestrationRuns()` - Run history
  - `getOrchestrationSummary()` - System metrics
  - `getWorkspaceOverview()` - Workspace info
  - `runManagedInstanceAction()` - Instance actions
- **Complex Features**:
  - Instance lifecycle tracking
  - Decision engine integration
  - Progress monitoring
  - Action dispatch system

#### 6. Instance Detail (`/runs/[instanceId]`)

- **File**: `app/runs/[instanceId]/page.tsx` → `components/instance-detail-view.tsx`
- **Purpose**: Deep dive into specific instance
- **Key State Variables**:
  - `detail: InstanceDetail | null` - Complete instance data
  - `loading: boolean` - Loading state
  - `error: string | null` - Error state
  - `notice: string | null` - Notifications
  - `stream: "stdout" | "stderr"` - Log stream selection
  - `busyAction: string | null` - Action in progress
- **API Connections**:
  - `getInstanceDetail()` - Detailed instance data (polls every 5s)
  - `runManagedInstanceAction()` - Instance actions
- **Complex Features**:
  - Real-time log streaming
  - Metrics trend visualization
  - Configuration snapshot inspection
  - Action availability management

#### 7. Monitoring (`/dashboard/monitoring`)

- **File**: `app/dashboard/monitoring/page.tsx`
- **Purpose**: Real-time system monitoring
- **Key State Variables**:
  - `instances: InstanceSummary[]` - All instances
  - `clusterNodes: ClusterNodeHardware[]` - Hardware status
  - `summary: OrchestrationSummary | null` - System summary
  - `loading: boolean` - Loading state
  - `filter: "all" | "running" | "completed" | "failed"` - Instance filter
  - `busyAction: string | null` - Action in progress
  - `lastRefresh: Date` - Refresh timestamp
  - `selectedId: string | null` - Selected instance for detail
- **API Connections**:
  - `getInstances()` - Instance list
  - `getOrchestrationSummary()` - System metrics
  - `getClusterNodes()` - Hardware status
  - `runManagedInstanceAction()` - Instance actions
- **Complex Features**:
  - Live instance monitoring with 6-second refresh
  - Cluster health visualization
  - Interactive instance inspection
  - Real-time metrics sparklines
  - Log streaming with syntax highlighting

#### 8. Datasets (`/datasets`)

- **File**: `app/datasets/page.tsx` → `components/datasets-view.tsx`
- **Purpose**: Dataset catalog and provenance
- **Key State Variables**:
  - Uses `useLabMetadata()` hook
  - `dashboard: DatasetDashboard | null` - Dataset metadata
  - `provenance: DatasetProvenance | null` - Lineage info
- **API Connections**:
  - `getDatasetDashboard()` - Dataset catalog
- **Complex Features**:
  - Provenance tracking
  - Pack manifest inspection
  - Lineage summary visualization

#### 9. Benchmarks (`/benchmarks`)

- **File**: `app/benchmarks/page.tsx` → `components/benchmarks-view.tsx`
- **Purpose**: Benchmark library browser
- **Key State Variables**:
  - Uses `useLabMetadata()` hook
- **API Connections**:
  - `getBenchmarks()` - Benchmark registry
- **Complex Features**:
  - Benchmark metadata display
  - Tag-based categorization

#### 10. Workspace (`/workspace`)

- **File**: `app/workspace/page.tsx` → `components/workspace-view.tsx`
- **Purpose**: Workspace configuration and capabilities
- **Key State Variables**:
  - `overview: WorkspaceOverview | null` - Workspace metadata
  - `loadError: string | null` - Error state
  - `notice: string | null` - Notifications
  - `loading: boolean` - Loading state
  - `copied: CopyMap` - Clipboard state
- **API Connections**:
  - `getWorkspaceOverview()` - Workspace information
- **Complex Features**:
  - Readiness checks display
  - Command recipe copying
  - Extension point visualization

### Dashboard Sub-Pages

#### Training (`/dashboard/training`)

- **File**: `app/dashboard/training/page.tsx`
- **Purpose**: Training workflow management
- **Status**: Placeholder implementation

#### Evaluation (`/dashboard/evaluate`)

- **File**: `app/dashboard/evaluate/page.tsx`
- **Purpose**: Evaluation workflow management
- **Status**: Placeholder implementation

#### Finetune (`/dashboard/finetune`)

- **File**: `app/dashboard/finetune/page.tsx`
- **Purpose**: Fine-tuning workflow management
- **Status**: Placeholder implementation

#### Deploy (`/dashboard/deploy`)

- **File**: `app/dashboard/deploy/page.tsx`
- **Purpose**: Deployment workflow management
- **Status**: Placeholder implementation

#### Inference (`/dashboard/inference`)

- **File**: `app/dashboard/inference/page.tsx`
- **Purpose**: Inference workflow management
- **Status**: Placeholder implementation

#### Datasets Lab (`/dashboard/datasets`)

- **File**: `app/dashboard/datasets/page.tsx`
- **Purpose**: Lab dataset management
- **Status**: Placeholder implementation

#### Agents Lab (`/dashboard/agents`)

- **File**: `app/dashboard/agents/page.tsx`
- **Purpose**: Agent swarm management
- **Status**: Placeholder implementation

#### AutoML Lab (`/dashboard/automl`)

- **File**: `app/dashboard/automl/page.tsx`
- **Purpose**: AutoML sweep management
- **Status**: Placeholder implementation

#### Cluster Lab (`/dashboard/cluster`)

- **File**: `app/dashboard/cluster/page.tsx`
- **Purpose**: Cluster resource management
- **Status**: Placeholder implementation

## API Layer Architecture

### Core API Client (`lib/api.ts`)

#### Base Configuration

- **Timeout**: 20 seconds per request
- **Retry Logic**: Falls back through multiple API base URLs
- **Error Handling**: Structured error extraction from FastAPI responses
- **Base URLs**: Configurable via `NEXT_PUBLIC_API_BASE_URL` or defaults

#### Primary API Endpoints

##### Generation & Comparison

```typescript
generateAnswer(payload: GenerateRequest): Promise<GenerateResponse>
compareModels(payload: CompareRequest): Promise<CompareResponse>
```

##### System Status

```typescript
getStatus(): Promise<StatusInfo>
getTitanStatus(): Promise<TitanStatus>
getWorkspaceOverview(): Promise<WorkspaceOverview>
```

##### Dataset Management

```typescript
getDatasetDashboard(): Promise<DatasetDashboard>
getPromptLibrary(): Promise<PromptLibrary>
synthesizeDataset(payload: SynthesizeRequest): Promise<SynthesizeResponse>
getSynthesisJob(jobId: string): Promise<SynthesisJob>
```

##### Model & Benchmark Registry

```typescript
getModels(): Promise<ModelInfo[]>
getBenchmarks(): Promise<BenchmarkInfo[]>
getRuns(): Promise<RunInfo[]>
```

##### Instance Management

```typescript
getInstances(): Promise<InstanceSummary[]>
getInstanceDetail(instanceId: string): Promise<InstanceDetail>
createManagedInstance(payload: CreateManagedInstanceRequest): Promise<InstanceDetail>
runManagedInstanceAction(instanceId: string, payload: ActionRequest): Promise<InstanceDetail>
evaluateManagedInstance(instanceId: string, payload?: EvalRequest): Promise<InstanceDetail>
startManagedInference(instanceId: string, payload?: InferenceRequest): Promise<InstanceDetail>
deployManagedInstance(instanceId: string, payload: DeployRequest): Promise<InstanceDetail>
```

##### Orchestration

```typescript
getOrchestrationRuns(): Promise<OrchestrationRun[]>
getOrchestrationRun(runId: string): Promise<OrchestrationRunDetail>
getOrchestrationSummary(): Promise<OrchestrationSummary>
getMissionControl(): Promise<MissionControlSnapshot>
```

##### Cluster & Hardware

```typescript
getClusterNodes(): Promise<ClusterNodeHardware[]>
```

##### Agent Management

```typescript
getAgentSwarmStatus(): Promise<AgentSwarmStatus[]>
getAgentLogs(limit?: number): Promise<AgentLogEvent[]>
deployAgent(payload: AgentDeployRequest): Promise<DeployResponse>
updateAgent(agentId: string, payload: AgentUpdateRequest): Promise<UpdateResponse>
```

##### Telemetry

```typescript
getTelemetryBacklog(): Promise<TelemetryRecord[]>
promoteTelemetryRecord(recordId: string): Promise<TelemetryActionResult>
discardTelemetryRecord(recordId: string): Promise<TelemetryActionResult>
flagTelemetry(payload: FlagTelemetryRequest): Promise<FlagResponse>
```

## State Management Patterns

### Custom Hooks

#### `useLabMetadata` (`hooks/use-lab-metadata.ts`)

- **Purpose**: Centralized metadata loading
- **State**: Datasets, prompts, models, benchmarks, runs, status
- **Pattern**: Parallel loading with Promise.allSettled
- **Error Handling**: Graceful fallbacks for failed endpoints
- **Refresh**: No auto-refresh (load-once pattern)

#### Local Component State

Most components use local `useState` for:

- Form inputs and UI controls
- Loading and error states
- Temporary selections and filters
- Action busy states

#### Global State Patterns

- **No global state management library** (Redux, Zustand, etc.)
- **Prop drilling** minimized through custom hooks
- **URL-based state** for routing and instance selection
- **localStorage** for user preferences (workspace mode/density)

### Data Flow Patterns

#### Polling vs Event-Driven

- **Mission Control**: 7.5-second polling
- **Instance Detail**: 5-second polling
- **Monitoring**: 6-second polling
- **Metadata**: Load-once (no polling)

#### Error Boundaries

- Component-level error handling
- Graceful degradation with fallback content
- User-friendly error messages
- Retry mechanisms where appropriate

## Complex State Variables

### Training & Instance State

```typescript
// Instance progress tracking
progress?: {
  stage: string;
  status_message?: string;
  percent?: number;
}

// Metrics collection
metrics_summary: Record<string, unknown>
metrics: {
  summary: Record<string, unknown>;
  points: MetricPoint[];
}

// Decision engine results
decision?: {
  action: string;
  rule: string;
  thresholds: Record<string, number>;
  explanation: string;
}

// Recommendation system
recommendations?: FeedbackRecommendation[]
```

### GPU & Hardware Metrics

```typescript
// Cluster node status
ClusterNodeHardware: {
  id: string;
  name: string;
  type: string;
  memory: string;
  usage: number;  // VRAM percentage
  status: "online" | "idle" | "offline";
  activeJobs: number;
}

// Titan hardware detection
TitanStatus: {
  backend: string;
  silicon: string;
  bandwidth_gbps?: number;
  supports_cuda: boolean;
  cuda_compute_capability?: string;
  remote_execution: boolean;
  cloud_provider?: string;
  preferred_training_backend: string;
}
```

### Training Logs & Telemetry

```typescript
// Live log streaming
logs?: {
  stdout: string;
  stderr: string;
  stdout_path?: string;
  stderr_path?: string;
}

// Telemetry backlog
TelemetryRecord: {
  id: string;
  timestamp: number;
  prompt: string;
  assistant_output: string;
  expected_output: string;
  model_variant: string;
  latency_s?: number;
}
```

### Orchestration State

```typescript
// AutoML sweep tracking
AutoMLSweep: {
  id: string;
  status: string;
  created_at: string;
  updated_at: string;
  config: Record<string, unknown>;
  results: Record<string, unknown>;
}

// Agent swarm status
AgentSwarmStatus: {
  id: string;
  name: string;
  role: string;
  model: string;
  status: 'active' | 'sleeping' | 'offline';
  uptime_s: number;
  tokens_used: number;
}
```

## Component Architecture

### Layout Components

- **AppShell**: Main layout wrapper
- **AppNav**: Navigation with active state
- **PageHeader**: Consistent page structure
- **StatePanel**: Loading/error/empty states

### Feature Components

- **ChatShell**: Interactive reasoning interface
- **CompareLab**: Model comparison workspace
- **RunsView**: Instance management
- **InstanceDetailView**: Deep instance inspection
- **MonitoringPage**: Real-time system monitoring
- **DatasetsView**: Dataset catalog
- **BenchmarksView**: Benchmark library
- **WorkspaceView**: Configuration and capabilities

### UI Components

- **MathBlock**: Mathematical expression rendering
- **ModelChip**: Model selection display
- **MetricBadge**: Metric visualization
- **CandidateInspector**: Answer candidate analysis
- **MetricsTrendChart**: Time-series visualization

## Data Types and Interfaces

### Core Domain Types

```typescript
// Instance lifecycle
LifecycleStage: 'prepare' | 'train' | 'evaluate' | 'decide' | 'finetune' | 'infer' | 'publish';
LearningMode: 'supervised' |
  'unsupervised' |
  'rlhf' |
  'dpo' |
  'orpo' |
  'ppo' |
  'lora' |
  'qlora' |
  'full_finetune';
DeploymentTarget: 'huggingface' |
  'ollama' |
  'lmstudio' |
  'api' |
  'openai_compatible_api' |
  'custom_api';

// Generation parameters
Difficulty: 'easy' | 'medium' | 'hard' | 'olympiad';
SolverMode: 'rigorous' | 'exam' | 'concise' | 'verification';
OutputFormat: 'text' | 'json';

// System state
InstanceStatus: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
OrchestrationStatus: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
```

## Performance Considerations

### Optimization Strategies

- **React.useTransition()** for non-blocking UI updates
- **Debounced API calls** for user inputs
- **Memoized selectors** for large data sets
- **Virtual scrolling** for long lists (not yet implemented)
- **Lazy loading** for heavy components

### Caching Patterns

- **No client-side caching library** (React Query, SWR)
- **Component-level caching** via useState
- **API-level caching** handled by backend
- **Browser cache** for static assets

## Security Considerations

### API Security

- **No authentication tokens** in frontend (assumes backend auth)
- **CORS handling** via Next.js proxy
- **Input validation** via Zod schemas
- **XSS prevention** via React's built-in protections

### Data Exposure

- **Sensitive configs** not exposed to frontend
- **Environment variables** for API endpoints
- **No direct database access** from frontend

## Accessibility & UX

### Responsive Design

- **Mobile-first** approach with Tailwind breakpoints
- **Touch-friendly** interface elements
- **Keyboard navigation** support
- **Screen reader** compatibility

### User Feedback

- **Loading states** for all async operations
- **Error boundaries** with recovery options
- **Success notifications** for user actions
- **Progress indicators** for long-running operations

## Future Architecture Considerations

### Potential Improvements

1. **Global State Management**: Consider Zustand/Jotai for complex state
2. **Data Fetching**: Implement React Query or SWR for caching
3. **Real-time Updates**: WebSocket integration for live data
4. **Performance**: Virtualization for large datasets
5. **Testing**: Component testing with React Testing Library
6. **Error Boundaries**: More sophisticated error handling
7. **Internationalization**: i18n support for multiple languages

### Scalability Concerns

- **Large dataset handling** in datasets view
- **Real-time metrics** at scale
- **Memory management** for long-running sessions
- **Network optimization** for API calls

---

This specification serves as the complete technical reference for the AI-Factory frontend, providing the foundation for architectural decisions, refactoring efforts, and feature development.
