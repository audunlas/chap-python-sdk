# Quick Start

This guide will help you get started with testing your chapkit models using the CHAP Python SDK.

## Basic Model Testing

The SDK provides utilities for testing models using the chapkit functional interface.

### Step 1: Define Your Model Functions

Models are defined as async functions following the chapkit functional interface:

```python
from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame


class MyModelConfig(BaseConfig):
    """Configuration for my model."""

    learning_rate: float = 0.01


async def my_train(config: BaseConfig, data: DataFrame, geo=None):
    """Train the model."""
    return {"means": 10.0}


async def my_predict(config: BaseConfig, model, historic: DataFrame, future: DataFrame, geo=None):
    """Generate predictions."""
    samples = [[model["means"]] * 10 for _ in range(len(future))]
    return DataFrame.from_dict({
        "time_period": list(future["time_period"]),
        "location": list(future["location"]),
        "samples": samples,
    })
```

### Step 2: Load Example Data

The SDK includes example datasets for testing:

```python
from chap_python_sdk.testing import get_example_data, list_available_datasets

# List available datasets
datasets = list_available_datasets()
# Returns: [("laos", "monthly"), ...]

# Load example data
example_data = get_example_data(country="laos", frequency="monthly")
```

### Step 3: Run Validation

Use `validate_model_io` to test your model:

```python notest
import pytest
from chap_python_sdk.testing import validate_model_io


@pytest.mark.asyncio
async def test_my_model():
    """Test my model against example data."""
    example_data = get_example_data(country="laos", frequency="monthly")
    config = MyModelConfig()

    result = await validate_model_io(my_train, my_predict, example_data, config)

    assert result.success, f"Validation failed: {result.errors}"
    assert result.n_predictions == 21
    assert result.n_samples >= 1
```

## Using FunctionalModelRunner

If you prefer to bundle your functions into a runner object:

```python notest
from chapkit import FunctionalModelRunner

runner = FunctionalModelRunner(on_train=my_train, on_predict=my_predict)
```

## Example Data Structure

The `ExampleData` object contains:

```python
example_data = get_example_data(country="laos", frequency="monthly")

# Access individual components
training_data = example_data.training_data      # Historical data for training
historic_data = example_data.historic_data      # Recent observations
future_data = example_data.future_data          # Future periods to predict
expected_predictions = example_data.predictions  # Reference predictions (optional)
```

## Next Steps

- Learn about [Model Testing](../user-guide/model-testing.md) in depth
- Explore [Test Data Generation](../user-guide/data-generation.md)
- Check out the [API Reference](../api/index.md)
