"""
Every tagged code snippet in the README must run and draw a chart.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docs"))
from make_figures import readme_examples, run_example  # noqa: E402

EXAMPLES = readme_examples()


def test_examples_found():
    assert len(EXAMPLES) >= 9


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_readme_example_runs(name):
    fig = run_example(EXAMPLES[name])
    assert fig.axes and fig.axes[0].get_title(loc="left")
