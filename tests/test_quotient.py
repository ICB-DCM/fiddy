import pytest

import fiddy
import numpy as np


@pytest.fixture
def line():
    def function(x):
        return (3 * x + 2)[0]

    def derivative(x):
        return 3

    return {
        "function": function,
        "derivative": derivative,
        "point": np.array([3]),
        "dimension": np.array([0]),
        "size": 1e-10,
    }


def test_forward(line):
    step = fiddy.step.dstep(
        point=line["point"],
        dimension=line["dimension"],
        size=line["size"],
    )
    fd = fiddy.quotient.forward(
        function=line["function"],
        point=line["point"],
        step=step,
    )
    expected = line["derivative"](line["point"])
    assert np.isclose(fd, expected)
