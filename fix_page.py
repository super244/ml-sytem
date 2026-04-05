with open("frontend/app/dashboard/page.tsx") as f:
    c = f.read()

c = c.replace("const instances = mission?.control_plane.instances ?? [];", "const instances = useMemo(() => mission?.control_plane.instances ?? [], [mission?.control_plane.instances]);")
c = c.replace("import { useEffect, useMemo, useState, useTransition } from 'react';", "import { useEffect, useMemo, useState, useTransition } from 'react';")

with open("frontend/app/dashboard/page.tsx", "w") as f:
    f.write(c)

