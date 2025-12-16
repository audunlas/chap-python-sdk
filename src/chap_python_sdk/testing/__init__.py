"""Testing utilities for chapkit model validation."""

from chapkit import FunctionalModelRunner

from chap_python_sdk.testing.assertions import (
    assert_consistent_sample_counts,
    assert_numeric_samples,
    assert_prediction_shape,
    assert_samples_column,
    assert_time_location_columns,
    assert_valid_predictions,
)
from chap_python_sdk.testing.example_data import (
    get_example_data,
    list_available_datasets,
)
from chap_python_sdk.testing.predictions import (
    detect_prediction_format,
    has_prediction_samples,
    predictions_from_long,
    predictions_from_wide,
    predictions_summary,
    predictions_to_long,
    predictions_to_quantiles,
    predictions_to_wide,
)
from chap_python_sdk.testing.types import (
    ExampleData,
    GeoFeatureCollection,
    PredictFunction,
    TrainFunction,
    ValidationResult,
)
from chap_python_sdk.testing.validation import (
    validate_model_io,
    validate_model_io_all,
)

__all__ = [
    # Example data
    "get_example_data",
    "list_available_datasets",
    # Validation
    "validate_model_io",
    "validate_model_io_all",
    # Assertions
    "assert_valid_predictions",
    "assert_prediction_shape",
    "assert_samples_column",
    "assert_consistent_sample_counts",
    "assert_numeric_samples",
    "assert_time_location_columns",
    # Predictions
    "predictions_to_wide",
    "predictions_from_wide",
    "predictions_to_long",
    "predictions_from_long",
    "detect_prediction_format",
    "has_prediction_samples",
    "predictions_to_quantiles",
    "predictions_summary",
    # Types
    "ExampleData",
    "GeoFeatureCollection",
    "ValidationResult",
    "TrainFunction",
    "PredictFunction",
    # Re-export from chapkit
    "FunctionalModelRunner",
]
