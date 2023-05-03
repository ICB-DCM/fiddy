# fiddy

Finite difference methods, for applications including gradient computation and gradient checks.

Install with `pip install -e .`.

# Important notes
The output of your function of interest should be a NumPy array. If your function is scalar-valued, change it to a NumPy array with:
```python
import numpy as np

def function(input_value: float) -> np.ndarray:
    scalar_output_value = old_function(input_value)
    return np.array([scalar_output_value])
```

# Installation
Currently under development, please install from source.
```bash
pip install -e .
```

Can also be installed from [PyPI](https://pypi.org/project/fiddy/0.0.1/)
```bash
pip install fiddy
```
