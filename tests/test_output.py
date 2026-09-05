import numpy as np

from fiddy import flatten_output
from fiddy.output import OutputSchema


def test_flatten_plain_array_roundtrips():
    array = np.array([[1.0, 2.0], [3.0, 4.0]])

    flat, schema = flatten_output(array)

    assert flat.shape == (4,)
    assert not schema.is_structured
    assert schema.size == 4

    restored = schema.unbundle(flat)
    np.testing.assert_array_equal(restored, array)


def test_flatten_dict_roundtrips_each_name():
    output = {
        "x": np.arange(6).reshape(2, 3),
        "y": np.array([10.0, 11.0]),
        "llh": np.array(42.0),
    }

    flat, schema = flatten_output(output)

    assert schema.is_structured
    assert schema.names == ["x", "y", "llh"]
    assert flat.shape == (schema.size,)
    assert schema.size == 6 + 2 + 1

    restored = schema.unbundle(flat)
    assert set(restored) == set(output)
    for name, array in output.items():
        np.testing.assert_array_equal(restored[name], array)
        assert restored[name].shape == array.shape


def test_unbundle_applies_to_any_array_with_the_same_layout():
    """`unbundle` is layout-only: it works on e.g. a computed derivative
    that shares the original output's flat layout, not just the original
    output values themselves."""
    output = {"x": np.zeros((2, 2)), "y": np.zeros(3)}
    _, schema = flatten_output(output)

    derivative_like = np.arange(schema.size, dtype=float)
    restored = schema.unbundle(derivative_like)

    np.testing.assert_array_equal(restored["x"], [[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_array_equal(restored["y"], [4.0, 5.0, 6.0])


def test_output_schema_size_handles_scalar_entries():
    schema = OutputSchema(
        names=["a"],
        shapes={"a": ()},
        slices={"a": slice(0, 1)},
        is_structured=False,
    )
    assert schema.size == 1
