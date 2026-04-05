with open("frontend/app/dashboard/automl/page.tsx") as f:
    c = f.read()

c = c.replace("  const modelOptions =\n    availableModels.length > 0\n      ? availableModels.map((model) => model.name)\n      : demoMode\n        ? MODELS\n        : [];", "  const modelOptions = useMemo(() => \n    availableModels.length > 0\n      ? availableModels.map((model) => model.name)\n      : demoMode\n        ? MODELS\n        : [], [availableModels, demoMode]);")

c = c.replace("import { useEffect, useState } from 'react';", "import { useEffect, useState, useMemo } from 'react';")
c = c.replace("  }, []);", "  }, [demoMode]);")

with open("frontend/app/dashboard/automl/page.tsx", "w") as f:
    f.write(c)

