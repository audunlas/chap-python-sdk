"""Validation and testing framework for chapkit models."""

from chap_python_sdk.testing import (
    ExampleData,
    FunctionalModelRunner,
    PredictFunction,
    TrainFunction,
    ValidationResult,
    assert_consistent_sample_counts,
    assert_numeric_samples,
    assert_prediction_shape,
    assert_samples_column,
    assert_time_location_columns,
    assert_valid_predictions,
    detect_prediction_format,
    get_example_data,
    has_prediction_samples,
    list_available_datasets,
    predictions_from_long,
    predictions_from_wide,
    predictions_summary,
    predictions_to_long,
    predictions_to_quantiles,
    predictions_to_wide,
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
    "ValidationResult",
    "TrainFunction",
    "PredictFunction",
    # Re-export from chapkit
    "FunctionalModelRunner",
]
