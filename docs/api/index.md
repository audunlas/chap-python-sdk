# API Reference

This is the complete API reference for the CHAP Python SDK.

## Modules

The SDK is organized into the following modules:

### [Types](types.md)

Core type definitions and data classes:

- `ExampleData` - Container for test datasets
- `RunInfo` - Runtime information for models
- `ValidationResult` - Validation outcome
- `TrainFunction` - Type alias for train functions
- `PredictFunction` - Type alias for predict functions

### [Validation](validation.md)

Model I/O validation:

- `validate_model_io()` - Main validation function
- `validate_model_io_all()` - Batch validation

### [Assertions](assertions.md)

Prediction validation assertions:

- `assert_valid_predictions()` - Comprehensive validation
- `assert_prediction_shape()` - Shape validation
- `assert_samples_column()` - Samples validation
- And more...

### [Predictions](predictions.md)

Prediction format utilities:

- `detect_prediction_format()` - Format detection
- `predictions_to_wide()` / `predictions_from_wide()` - Wide format conversion
- `predictions_to_long()` / `predictions_from_long()` - Long format conversion
- `predictions_to_quantiles()` - Quantile generation
- `predictions_summary()` - Summary statistics

### [Generators](generators.md)

Test data generation:

- `MLServiceInfo` - Model requirements specification
- `DataGenerationConfig` - Generation configuration
- `PeriodType` - Time period frequency enum
- `generate_test_data()` - Generate test data
- `generate_run_info()` - Generate RunInfo
- `generate_example_data()` - Generate ExampleData

### [Example Data](example-data.md)

Bundled example datasets:

- `list_available_datasets()` - List available datasets
- `get_example_data()` - Load example data

## Import Patterns

All public API is available from the main testing module:

```python
from chap_python_sdk.testing import (
    # Types
    ExampleData,
    RunInfo,
    ValidationResult,
    TrainFunction,
    PredictFunction,

    # Validation
    validate_model_io,
    validate_model_io_all,

    # Assertions
    assert_valid_predictions,
    assert_prediction_shape,
    assert_samples_column,
    PredictionValidationError,

    # Predictions
    detect_prediction_format,
    predictions_to_wide,
    predictions_from_wide,
    predictions_to_long,
    predictions_from_long,

    # Generators
    MLServiceInfo,
    DataGenerationConfig,
    PeriodType,
    generate_test_data,

    # Example Data
    list_available_datasets,
    get_example_data,
)
```
