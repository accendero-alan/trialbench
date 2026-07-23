"""Method registry and implementations.

Importing this package registers every available method into the global
registry. Heavy optional dependencies are imported lazily inside each method's
``fit`` so that a missing dependency never breaks the core run.
"""
from . import classical  # noqa: F401  (Tier A: sklearn — always available)
from . import gbm  # noqa: F401        (Tier A: xgboost/lightgbm/catboost)
from . import text_nlp  # noqa: F401   (Tier A/C: tf-idf + clinical embeddings)
from . import deep_tabular  # noqa: F401  (Tier B stubs)
from . import multimodal  # noqa: F401    (Tier D stubs)
from . import llm  # noqa: F401          (Tier D stub)
