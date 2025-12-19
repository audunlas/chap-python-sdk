"""Assertion helpers for model I/O validation."""

from numbers import Number
from typing import Any, cast

import numpy as np
from chapkit.data import DataFrame


def _get_column(dataframe: DataFrame, column: str) -> list[Any]:
    """Get a column from a DataFrame as a list."""
    return cast(list[Any], dataframe[column])


class PredictionValidationError(AssertionError):
    """Error raised when prediction validation fails."""

    pass


def assert_valid_predictions(
    predictions: DataFrame,
    expected_rows: int | None = None,
) -> None:
    """Assert predictions have valid structure.

    Args:
        predictions: DataFrame to validate.
        expected_rows: Expected number of rows (optional).

    Raises:
        PredictionValidationError: If validation fails.
    """
    if not isinstance(predictions, DataFrame):
        raise PredictionValidationError(f"Predictions must be a DataFrame, got {type(predictions).__name__}")

    if len(predictions) == 0:
        raise PredictionValidationError("Predictions DataFrame is empty")

    if expected_rows is not None and len(predictions) != expected_rows:
        raise PredictionValidationError(f"Expected {expected_rows} prediction rows, got {len(predictions)}")

    # Check for required columns
    assert_time_location_columns(predictions)
    assert_samples_column(predictions)
    assert_consistent_sample_counts(predictions)
    assert_numeric_samples(predictions)


def assert_prediction_shape(
    predictions: DataFrame,
    future_data: DataFrame,
) -> None:
    """Assert predictions match expected shape from future_data.

    Args:
        predictions: Predictions DataFrame.
        future_data: Future data DataFrame that predictions should match.

    Raises:
        PredictionValidationError: If shape doesn't match.
    """
    if len(predictions) != len(future_data):
        raise PredictionValidationError(
            f"Predictions have {len(predictions)} rows, but future_data has {len(future_data)} rows"
        )

    # Check that time_period and location values match
    if "time_period" in predictions.columns and "time_period" in future_data.columns:
        pred_periods = set(predictions["time_period"])
        future_periods = set(future_data["time_period"])
        if pred_periods != future_periods:
            missing = future_periods - pred_periods
            extra = pred_periods - future_periods
            message = "Time periods don't match between predictions and future_data."
            if missing:
                message += f" Missing: {missing}."
            if extra:
                message += f" Extra: {extra}."
            raise PredictionValidationError(message)

    if "location" in predictions.columns and "location" in future_data.columns:
        pred_locations = set(predictions["location"])
        future_locations = set(future_data["location"])
        if pred_locations != future_locations:
            missing = future_locations - pred_locations
            extra = pred_locations - future_locations
            message = "Locations don't match between predictions and future_data."
            if missing:
                message += f" Missing: {missing}."
            if extra:
                message += f" Extra: {extra}."
            raise PredictionValidationError(message)


def assert_samples_column(
    predictions: DataFrame,
    min_samples: int = 1,
    max_samples: int | None = None,
) -> None:
    """Assert samples column contains valid numeric lists.

    Args:
        predictions: DataFrame to validate.
        min_samples: Minimum number of samples required per row.
        max_samples: Maximum number of samples allowed per row (optional).

    Raises:
        PredictionValidationError: If validation fails.
    """
    if "samples" not in predictions.columns:
        raise PredictionValidationError("Predictions must have 'samples' column")

    samples_column = _get_column(predictions, "samples")
    for idx in range(len(predictions)):
        samples = samples_column[idx]

        if not isinstance(samples, (list, np.ndarray)):
            raise PredictionValidationError(f"Row {idx}: 'samples' must be a list, got {type(samples).__name__}")

        n_samples = len(samples)
        if n_samples < min_samples:
            raise PredictionValidationError(f"Row {idx}: Expected at least {min_samples} samples, got {n_samples}")

        if max_samples is not None and n_samples > max_samples:
            raise PredictionValidationError(f"Row {idx}: Expected at most {max_samples} samples, got {n_samples}")


def assert_consistent_sample_counts(predictions: DataFrame) -> None:
    """Assert all rows have the same number of samples.

    Args:
        predictions: DataFrame to validate.

    Raises:
        PredictionValidationError: If sample counts are inconsistent.
    """
    if "samples" not in predictions.columns:
        raise PredictionValidationError("Predictions must have 'samples' column")

    if len(predictions) == 0:
        return

    samples_column = _get_column(predictions, "samples")
    sample_counts = [len(samples_column[idx]) for idx in range(len(predictions))]
    unique_counts = set(sample_counts)

    if len(unique_counts) > 1:
        raise PredictionValidationError(
            f"Inconsistent sample counts across rows: {unique_counts}. All rows must have the same number of samples."
        )


def assert_numeric_samples(predictions: DataFrame) -> None:
    """Assert all sample values are numeric.

    Args:
        predictions: DataFrame to validate.

    Raises:
        PredictionValidationError: If non-numeric samples are found.
    """
    if "samples" not in predictions.columns:
        raise PredictionValidationError("Predictions must have 'samples' column")

    samples_column = _get_column(predictions, "samples")
    for idx in range(len(predictions)):
        samples = samples_column[idx]

        for sample_idx, value in enumerate(samples):
            if not isinstance(value, Number) and not (
                isinstance(value, np.generic) and np.issubdtype(value.dtype, np.number)
            ):
                raise PredictionValidationError(
                    f"Row {idx}, sample {sample_idx}: Expected numeric value, got {type(value).__name__}"
                )

            # Check for NaN/Inf
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                raise PredictionValidationError(f"Row {idx}, sample {sample_idx}: Sample value is {value}")


def assert_time_location_columns(predictions: DataFrame) -> None:
    """Assert predictions have time_period and location columns.

    Args:
        predictions: DataFrame to validate.

    Raises:
        PredictionValidationError: If required columns are missing.
    """
    missing_columns = []

    if "time_period" not in predictions.columns:
        missing_columns.append("time_period")

    if "location" not in predictions.columns:
        missing_columns.append("location")

    if missing_columns:
        raise PredictionValidationError(f"Predictions missing required columns: {missing_columns}")


def assert_wide_format_predictions(predictions: DataFrame) -> None:
    """Assert predictions are in wide format (sample_0, sample_1, ...).

    Args:
        predictions: DataFrame to validate.

    Raises:
        PredictionValidationError: If not in wide format.
    """
    sample_columns = [col for col in predictions.columns if col.startswith("sample_")]
    if not sample_columns:
        raise PredictionValidationError(
            "Predictions must have sample columns (sample_0, sample_1, ...). "
            "Found columns: " + ", ".join(predictions.columns)
        )


def assert_nonnegative_predictions(predictions: DataFrame) -> None:
    """Assert all prediction values are non-negative.

    Works with both wide format (sample_0, sample_1, ...) and nested format (samples column).

    Args:
        predictions: DataFrame to validate.

    Raises:
        PredictionValidationError: If negative values are found.
    """
    # Check for wide format (sample_0, sample_1, ...)
    sample_columns = [col for col in predictions.columns if col.startswith("sample_")]

    if sample_columns:
        for col in sample_columns:
            values = _get_column(predictions, col)
            for idx, val in enumerate(values):
                if isinstance(val, (int, float)) and val < 0:
                    raise PredictionValidationError(
                        f"Row {idx}, {col}: Negative value {val} not allowed. Predictions must be non-negative."
                    )
    elif "samples" in predictions.columns:
        # Check nested format
        samples_column = _get_column(predictions, "samples")
        for idx in range(len(predictions)):
            samples = samples_column[idx]
            for sample_idx, val in enumerate(samples):
                if isinstance(val, (int, float)) and val < 0:
                    raise PredictionValidationError(
                        f"Row {idx}, sample {sample_idx}: Negative value {val} not allowed. "
                        "Predictions must be non-negative."
                    )
    else:
        raise PredictionValidationError("Predictions must have sample columns or 'samples' column")


def assert_no_nan_predictions(predictions: DataFrame) -> None:
    """Assert no NaN values in predictions.

    Works with both wide format (sample_0, sample_1, ...) and nested format (samples column).

    Args:
        predictions: DataFrame to validate.

    Raises:
        PredictionValidationError: If NaN values are found.
    """
    # Check for wide format (sample_0, sample_1, ...)
    sample_columns = [col for col in predictions.columns if col.startswith("sample_")]

    if sample_columns:
        for col in sample_columns:
            values = _get_column(predictions, col)
            for idx, val in enumerate(values):
                if isinstance(val, float) and np.isnan(val):
                    raise PredictionValidationError(f"Row {idx}, {col}: NaN value not allowed")
    elif "samples" in predictions.columns:
        # Check nested format
        samples_column = _get_column(predictions, "samples")
        for idx in range(len(predictions)):
            samples = samples_column[idx]
            for sample_idx, val in enumerate(samples):
                if isinstance(val, float) and np.isnan(val):
                    raise PredictionValidationError(f"Row {idx}, sample {sample_idx}: NaN value not allowed")
    else:
        raise PredictionValidationError("Predictions must have sample columns or 'samples' column")
