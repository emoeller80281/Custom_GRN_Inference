"""Command-line runner that wires the per-section child classes together.

Each analysis section is a child of :class:`~.base.TFTGBase` living in its own
module. This runner lets you execute one, several, or all sections:

    python -m test_tf_tg_predictions.run                 # run everything
    python -m test_tf_tg_predictions.run generalizability grn_sizes
    python -m test_tf_tg_predictions.run --list

Sections are heavy (they load models / cached datasets and write figures under
``plots/``), so run them individually while iterating. This mirrors the original
notebook, which is kept unmodified as the canonical exploratory version.
"""
import argparse

from .model_vs_testset import ModelVsTestSet
from .generalizability import Generalizability
from .cross_model import CrossModelComparison
from .training_gif import TrainingHistogramGIF
from .tf_dna import TFDNAModelEvaluation
from .grn_sizes import GRNSizeComparison
from .stability import StabilityAnalysis
from .feature_ablation import FeatureAblation

# Ordered as in the notebook.
SECTIONS = {
    "model_vs_testset": ModelVsTestSet,
    "generalizability": Generalizability,
    "cross_model": CrossModelComparison,
    "training_gif": TrainingHistogramGIF,
    "tf_dna": TFDNAModelEvaluation,
    "grn_sizes": GRNSizeComparison,
    "stability": StabilityAnalysis,
    "feature_ablation": FeatureAblation,
}


def run_sections(names):
    """Instantiate and ``run()`` each named section in order."""
    for name in names:
        cls = SECTIONS[name]
        print(f"\n=== Running section: {name} ({cls.__name__}) ===", flush=True)
        cls().run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "sections", nargs="*", metavar="SECTION",
        help="Section(s) to run. Choices: " + ", ".join(SECTIONS)
             + ". Default: all.",
    )
    parser.add_argument("--list", action="store_true", help="List available sections and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for name, cls in SECTIONS.items():
            print(f"{name:20s} -> {cls.__name__}")
        return

    if not args.sections or "all" in args.sections:
        names = list(SECTIONS)
    else:
        unknown = [s for s in args.sections if s not in SECTIONS]
        if unknown:
            parser.error(f"Unknown section(s): {unknown}. Choices: {list(SECTIONS)}")
        names = args.sections

    run_sections(names)


if __name__ == "__main__":
    main()
