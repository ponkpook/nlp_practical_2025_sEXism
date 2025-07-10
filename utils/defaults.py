from yacs.config import CfgNode as CN

# Create a root configuration node.
cfg = CN()

# -----------------------------------------------------------------------------
# Basic Project Settings
# -----------------------------------------------------------------------------
cfg.out_dir = 'results/'
cfg.dataset = 'default_dataset.csv'
cfg.seed = 42
cfg.task = 0.0
cfg.dataset = CN()
cfg.dataset.path = ''
cfg.dataset.language = 'en'
cfg.dataset.columns = CN()
cfg.dataset.columns.text = 'text'
cfg.dataset.columns.target = 'sentiment'
cfg.dataset.eval_path = None

# -----------------------------------------------------------------------------
# Preprocessing Settings (Opt-in List)
# -----------------------------------------------------------------------------
# An ordered list of preprocessing steps. The user's YAML file will populate
# this list. If the list is empty, no preprocessing is performed.
cfg.preprocessing = []

# -----------------------------------------------------------------------------
# Feature Extraction Settings (Opt-in Dictionary)
# -----------------------------------------------------------------------------
# An empty container. The user's YAML will add named feature extractors here.
# For example, `feature_extraction.count_vectorize: ...` in YAML will add it.
cfg.feature_extraction = CN()
cfg.feature_extraction.count_vectorize = CN()
cfg.feature_extraction.count_vectorize.use = False  # Default to disabled
cfg.feature_extraction.count_vectorize.ngram_range = [1, 1]

# -----------------------------------------------------------------------------
# Model Settings
# -----------------------------------------------------------------------------
cfg.model = CN()
cfg.model.type = 'LogisticRegression'
# An empty container for model-specific parameters, populated by YAML.
cfg.model.params = CN(new_allowed=True)
cfg.model.layers = []

# -----------------------------------------------------------------------------
# Training Settings
# -----------------------------------------------------------------------------
cfg.train = CN()
cfg.train.epochs = 10
cfg.train.batch_size = 32
cfg.train.lr = 0.001

# -----------------------------------------------------------------------------
# Optimizer Settings
# -----------------------------------------------------------------------------
cfg.optimizer = CN()
cfg.optimizer.type = 'AdamW'
cfg.optimizer.params = CN(new_allowed=True)

# -----------------------------------------------------------------------------
# Scheduler Settings
# -----------------------------------------------------------------------------
cfg.scheduler = CN()
cfg.scheduler.type = 'StepLR'
cfg.scheduler.params = CN(new_allowed=True)


def get_cfg_defaults():
  """
  Returns a clone of the default configuration node.
  This prevents any changes to the default config from affecting other runs.
  """
  return cfg.clone()