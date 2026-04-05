from __future__ import annotations


def fix_file(filepath: str) -> None:
    with open(filepath) as f:
        content = f.read()

    # Replace local storage effects
    content = content.replace("setWorkspaceMode(storedMode);", "setTimeout(() => setWorkspaceMode(storedMode), 0);")
    content = content.replace("setDensity(storedDensity);", "setTimeout(() => setDensity(storedDensity), 0);")

    # For chat-shell.tsx and compare-lab.tsx, replace the sync effects with render-phase state sync

    # 1. modelVariant
    content = content.replace(
        """  useEffect(() => {
    setModelVariant((current) => pickPrimaryModel(availableModels, current));
  }, [availableModels]);""",
        """  const availableModelsStr = JSON.stringify(availableModels);
  const [prevModelsStr, setPrevModelsStr] = useState(availableModelsStr);
  if (availableModelsStr !== prevModelsStr) {
    setPrevModelsStr(availableModelsStr);
    setModelVariant(pickPrimaryModel(availableModels, modelVariant));
  }""",
    )

    # 2. compareToModel
    content = content.replace(
        """  useEffect(() => {
    setCompareToModel((current) => pickSecondaryModel(availableModels, modelVariant, current));
  }, [availableModels, modelVariant]);""",
        """  const [prevCompareVariant, setPrevCompareVariant] = useState(modelVariant);
  if (availableModelsStr !== prevModelsStr || modelVariant !== prevCompareVariant) {
    setPrevCompareVariant(modelVariant);
    setCompareToModel(pickSecondaryModel(availableModels, modelVariant, compareToModel));
  }""",
    )

    # 3. promptPreset
    content = content.replace(
        """  useEffect(() => {
    setPromptPreset((current) => pickPromptPreset(promptPresets, ['atlas_rigorous'], current));
  }, [promptPresets]);""",
        """  const promptPresetsStr = JSON.stringify(promptPresets);
  const [prevPresetsStr, setPrevPresetsStr] = useState(promptPresetsStr);
  if (promptPresetsStr !== prevPresetsStr) {
    setPrevPresetsStr(promptPresetsStr);
    setPromptPreset(pickPromptPreset(promptPresets, ['atlas_rigorous'], promptPreset));
  }""",
    )

    # Same for primaryModel and secondaryModel in compare-lab.tsx
    content = content.replace(
        """  useEffect(() => {
    setPrimaryModel((current) => pickPrimaryModel(models, current));
  }, [models]);""",
        """  const modelsStr = JSON.stringify(models);
  const [prevModelsStr, setPrevModelsStr] = useState(modelsStr);
  if (modelsStr !== prevModelsStr) {
    setPrevModelsStr(modelsStr);
    setPrimaryModel(pickPrimaryModel(models, primaryModel));
  }""",
    )

    content = content.replace(
        """  useEffect(() => {
    setSecondaryModel((current) => pickSecondaryModel(models, primaryModel, current));
  }, [models, primaryModel]);""",
        """  const [prevCompareVariant, setPrevCompareVariant] = useState(primaryModel);
  if (modelsStr !== prevModelsStr || primaryModel !== prevCompareVariant) {
    setPrevCompareVariant(primaryModel);
    setSecondaryModel(pickSecondaryModel(models, primaryModel, secondaryModel));
  }""",
    )

    with open(filepath, "w") as f:
        f.write(content)


fix_file("frontend/components/chat-shell.tsx")
fix_file("frontend/components/compare-lab.tsx")
