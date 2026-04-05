with open("frontend/app/dashboard/inference/page.tsx") as f:
    c = f.read()

c = c.replace(
    "const availableModels = models.length ? models : demoMode ? FALLBACK_MODELS : [];",
    "const availableModels = useMemo(() => models.length ? models : demoMode ? FALLBACK_MODELS : [], [models, demoMode]);",
)

c = c.replace(
    "const availablePromptPresets = promptPresets.length\n    ? promptPresets\n    : demoMode\n      ? FALLBACK_PROMPTS\n      : [];",
    "const availablePromptPresets = useMemo(() => promptPresets.length\n    ? promptPresets\n    : demoMode\n      ? FALLBACK_PROMPTS\n      : [], [promptPresets, demoMode]);",
)

c = c.replace(
    "import { useEffect, useRef, useState, useTransition } from 'react';",
    "import { useEffect, useRef, useState, useTransition, useMemo } from 'react';",
)

with open("frontend/app/dashboard/inference/page.tsx", "w") as f:
    f.write(c)
