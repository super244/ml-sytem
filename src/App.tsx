import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes, Navigate } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import Dashboard from "./pages/Dashboard";
import Monitoring from "./pages/Monitoring";
import ModelRegistry from "./pages/ModelRegistry";
import InferenceChat from "./pages/InferenceChat";
import DatasetStudio from "./pages/DatasetStudio";
import AutoMLExplorer from "./pages/AutoMLExplorer";
import AgentMonitor from "./pages/AgentMonitor";
import ClusterPage from "./pages/ClusterPage";
import Solve from "./pages/Solve";
import SettingsPage from "./pages/SettingsPage";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/dashboard/monitoring" element={<Monitoring />} />
          <Route path="/models" element={<ModelRegistry />} />
          <Route path="/inference" element={<InferenceChat />} />
          <Route path="/lab/datasets" element={<DatasetStudio />} />
          <Route path="/lab/automl" element={<AutoMLExplorer />} />
          <Route path="/lab/agents" element={<AgentMonitor />} />
          <Route path="/lab/cluster" element={<ClusterPage />} />
          <Route path="/solve" element={<Solve />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
