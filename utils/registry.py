class Registry:
    """A registry to map strings to classes or functions."""
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, name):
        def decorator(func_or_class):
            self._module_dict[name] = func_or_class
            return func_or_class
        return decorator

    def get(self, name):
        print(self._module_dict)
        if name not in self._module_dict:
            raise KeyError(f"'{name}' is not registered in '{self._name}' registry")
        return self._module_dict[name]

# --- Create the registry instances ---
PREPROCESSING_STEPS = Registry("Preprocessing")
FEATURE_EXTRACTORS = Registry("Features")
MODELS = Registry("Models")
print("--- REGISTRY.PY: The 'MODELS' registry instance has been created. ---") # <-- ADD THIS



# --- Create the clean, user-facing decorator aliases ---
register_preprocessing = PREPROCESSING_STEPS.register
register_feature_extractor = FEATURE_EXTRACTORS.register
register_model = MODELS.register
