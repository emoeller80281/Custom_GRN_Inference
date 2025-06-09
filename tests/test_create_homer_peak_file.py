import os
import sys
import pandas as pd
import pytest

# Ensure the src directory is on the Python path so grn_inference can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from grn_inference.create_homer_peak_file import convert_to_homer_peak_format


def test_convert_well_formed_peak_ids():
    df = pd.DataFrame({"peak_id": ["chr1:100-200", "chr2:300-400"]})
    result = convert_to_homer_peak_format(df)
    expected = pd.DataFrame({
        "peak_id": ["peak1", "peak2"],
        "chromosome": ["chr1", "chr2"],
        "start": [100, 300],
        "end": [200, 400],
        "strand": [".", "."],
    })
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_convert_raises_on_malformed_peak_ids():
    df = pd.DataFrame({"peak_id": ["chr1:100-200", "invalid"]})
    with pytest.raises(ValueError):
        convert_to_homer_peak_format(df)
