"""Class-based refactor of the ``test_tf_tg_predictions.ipynb`` notebook.

The notebook is kept as the canonical, unmodified exploratory version. This
package splits the same analysis into a parent class (:class:`~.base.TFTGBase`)
that holds the shared configuration (paths, formatting dictionaries, fonts,
model checkpoints) and the shared "main" functions, plus one child class per
section of related plots. Each child is responsible for the data generation,
caching, and plotting of its section.

Typical use::

    from test_tf_tg_predictions import Generalizability
    Generalizability().run()

or from the command line::

    python -m test_tf_tg_predictions.run generalizability
"""

from .base import TFTGBase
from .model_vs_testset import ModelVsTestSet
from .generalizability import Generalizability
from .cross_model import CrossModelComparison
from .training_gif import TrainingHistogramGIF
from .tf_dna import TFDNAModelEvaluation
from .grn_sizes import GRNSizeComparison
from .stability import StabilityAnalysis
from .feature_ablation import FeatureAblation

__all__ = [
    "TFTGBase",
    "ModelVsTestSet",
    "Generalizability",
    "CrossModelComparison",
    "TrainingHistogramGIF",
    "TFDNAModelEvaluation",
    "GRNSizeComparison",
    "StabilityAnalysis",
    "FeatureAblation",
]
