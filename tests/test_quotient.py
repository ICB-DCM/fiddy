import pytest

import finite_difference_methods as fdm
import numpy as np


@pytest.fixture
def line():
    def function(x):
        return (3*x + 2)[0]
    def derivative(x):
        return 3
    return {
        'function': function,
        'derivative': derivative,
        'point': np.array([3]),
        'dimension': np.array([0]),
        'size': 1e-10,
    }


def test_forward(line):
    step = fdm.step.dstep(
        point=line['point'],
        dimension=line['dimension'],
        size=line['size'],
    )
    fd = fdm.quotient.forward(
        function=line['function'],
        point=line['point'],
        step=step,
    )
    expected = line['derivative'](line['point'])
    assert np.isclose(fd, expected)
