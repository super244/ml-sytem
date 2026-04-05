with open("frontend/app/dashboard/monitoring/page.tsx") as f:
    c = f.read()

c = c.replace("const lines = logText\n    ? logText\n        .split('\\n')\n        .filter((line) => line.trim())\n        .slice(-200)\n    : ['No log output yet — instance is initializing…'];", "const lines = useMemo(() => logText\n    ? logText\n        .split('\\n')\n        .filter((line) => line.trim())\n        .slice(-200)\n    : ['No log output yet — instance is initializing…'], [logText]);")

c = c.replace("import { useEffect, useRef, useState } from 'react';", "import { useEffect, useRef, useState, useMemo } from 'react';")

with open("frontend/app/dashboard/monitoring/page.tsx", "w") as f:
    f.write(c)

