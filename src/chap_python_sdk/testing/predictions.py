"""Prediction format utilities for converting between different formats."""

import re
from typing import Any, Literal, cast

import numpy as np
from chapkit.data import DataFrame


def _get_column(dataframe: DataFrame, column: str) -> list[Any]:
    """Get a column from a DataFrame as a list."""
    return cast(list[Any], dataframe[column])


# Column names that identify non-sample data columns
NON_SAMPLE_COLUMNS = {"time_period", "location", "date", "sample_id", "prediction"}

# Pattern for wide format sample columns
SAMPLE_COLUMN_PATTERN = re.compile(r"^sample_(\d+)$")


def detect_prediction_format(dataframe: DataFrame) -> Literal["nested", "wide", "long"]:
    """Detect the prediction format of a DataFrame.

    Args:
        dataframe: DataFrame to analyze.

    Returns:
        Format type: "nested", "wide", or "long".

    Raises:
        ValueError: If the format cannot be determined.
    """
    columns = set(dataframe.columns)

    # Check for nested format (has 'samples' column)
    if "samples" in columns:
        return "nested"

    # Check for wide format (has sample_0, sample_1, ... columns)
    sample_columns = [col for col in columns if SAMPLE_COLUMN_PATTERN.match(col)]
    if sample_columns:
        return "wide"

    # Check for long format (has 'sample_id' and 'prediction' columns)
    if "sample_id" in columns and "prediction" in columns:
        return "long"

    raise ValueError(
        "Cannot determine prediction format. Expected one of: "
        "'samples' column (nested), 'sample_N' columns (wide), "
        "or 'sample_id'+'prediction' columns (long)."
    )


def has_prediction_samples(predictions: DataFrame) -> bool:
    """Check if DataFrame has valid samples column.

    Args:
        predictions: DataFrame to check.

    Returns:
        True if predictions has a valid 'samples' column with list values.
    """
    if "samples" not in predictions.columns:
        return False

    # Check that samples column contains lists
    samples_column = _get_column(predictions, "samples")
    if len(samples_column) == 0:
        return True

    first_value = samples_column[0]
    return isinstance(first_value, (list, np.ndarray))


def predictions_to_wide(predictions: DataFrame) -> DataFrame:
    """Convert nested format to wide format.

    Args:
        predictions: DataFrame with 'samples' column containing lists.

    Returns:
        DataFrame with sample_0, sample_1, ... columns.

    Raises:
        ValueError: If predictions is not in nested format.
    """
    if "samples" not in predictions.columns:
        raise ValueError("Predictions must have 'samples' column for nested format.")

    # Get non-sample columns
    non_sample_columns = [col for col in predictions.columns if col != "samples"]

    # Extract samples and determine number of samples
    samples_list = _get_column(predictions, "samples")
    if not samples_list:
        return predictions.drop(columns=["samples"])

    n_samples = len(samples_list[0])

    # Create wide format data
    data = {col: list(predictions[col]) for col in non_sample_columns}
    for i in range(n_samples):
        data[f"sample_{i}"] = [row[i] if i < len(row) else None for row in samples_list]

    return DataFrame.from_dict(data)


def predictions_from_wide(predictions: DataFrame) -> DataFrame:
    """Convert wide format to nested format.

    Args:
        predictions: DataFrame with sample_0, sample_1, ... columns.

    Returns:
        DataFrame with 'samples' column containing lists.

    Raises:
        ValueError: If predictions is not in wide format.
    """
    columns = list(predictions.columns)
    sample_columns = sorted(
        [col for col in columns if SAMPLE_COLUMN_PATTERN.match(col)],
        key=lambda x: int(SAMPLE_COLUMN_PATTERN.match(x).group(1)),  # type: ignore[union-attr]
    )

    if not sample_columns:
        raise ValueError("Predictions must have sample_N columns for wide format.")

    non_sample_columns = [col for col in columns if col not in sample_columns]

    # Create nested format data
    data = {col: list(predictions[col]) for col in non_sample_columns}

    # Combine sample columns into lists
    samples = []
    for idx in range(len(predictions)):
        row_samples = [_get_column(predictions, col)[idx] for col in sample_columns]
        samples.append(row_samples)

    data["samples"] = samples
    return DataFrame.from_dict(data)


def predictions_to_long(predictions: DataFrame) -> DataFrame:
    """Convert nested format to long format.

    Args:
        predictions: DataFrame with 'samples' column containing lists.

    Returns:
        DataFrame with 'sample_id' and 'prediction' columns.

    Raises:
        ValueError: If predictions is not in nested format.
    """
    if "samples" not in predictions.columns:
        raise ValueError("Predictions must have 'samples' column for nested format.")

    non_sample_columns = [col for col in predictions.columns if col != "samples"]

    rows = []
    for idx in range(len(predictions)):
        base_row = {col: _get_column(predictions, col)[idx] for col in non_sample_columns}
        samples = _get_column(predictions, "samples")[idx]

        for sample_id, prediction in enumerate(samples):
            row = base_row.copy()
            row["sample_id"] = sample_id
            row["prediction"] = prediction
            rows.append(row)

    return DataFrame.from_records(rows)


def predictions_from_long(predictions: DataFrame) -> DataFrame:
    """Convert long format to nested format.

    Args:
        predictions: DataFrame with 'sample_id' and 'prediction' columns.

    Returns:
        DataFrame with 'samples' column containing lists.

    Raises:
        ValueError: If predictions is not in long format.
    """
    if "sample_id" not in predictions.columns or "prediction" not in predictions.columns:
        raise ValueError("Predictions must have 'sample_id' and 'prediction' columns for long format.")

    non_sample_columns = [col for col in predictions.columns if col not in {"sample_id", "prediction"}]

    # Group by non-sample columns
    grouped: dict[tuple[object, ...], list[tuple[int, float]]] = {}
    for idx in range(len(predictions)):
        key = tuple(_get_column(predictions, col)[idx] for col in non_sample_columns)
        sample_id = int(_get_column(predictions, "sample_id")[idx])
        prediction = float(_get_column(predictions, "prediction")[idx])

        if key not in grouped:
            grouped[key] = []
        grouped[key].append((sample_id, prediction))

    # Build result
    rows = []
    for key, samples in grouped.items():
        row = dict(zip(non_sample_columns, key, strict=True))
        # Sort by sample_id and extract predictions
        sorted_samples = sorted(samples, key=lambda x: x[0])
        row["samples"] = [s[1] for s in sorted_samples]
        rows.append(row)

    return DataFrame.from_records(rows)


def predictions_to_quantiles(
    predictions: DataFrame,
    probabilities: list[float] | None = None,
) -> DataFrame:
    """Compute quantiles from prediction samples.

    Args:
        predictions: DataFrame with 'samples' column.
        probabilities: Quantile probabilities (default: [0.025, 0.25, 0.5, 0.75, 0.975]).

    Returns:
        DataFrame with quantile columns (q_0.025, q_0.25, etc.).
    """
    if probabilities is None:
        probabilities = [0.025, 0.25, 0.5, 0.75, 0.975]

    if "samples" not in predictions.columns:
        raise ValueError("Predictions must have 'samples' column.")

    non_sample_columns = [col for col in predictions.columns if col != "samples"]
    data = {col: list(predictions[col]) for col in non_sample_columns}

    samples_column = _get_column(predictions, "samples")
    for prob in probabilities:
        quantile_values = []
        for idx in range(len(predictions)):
            samples = samples_column[idx]
            quantile_values.append(float(np.quantile(samples, prob)))
        data[f"q_{prob}"] = quantile_values

    return DataFrame.from_dict(data)


def predictions_summary(
    predictions: DataFrame,
    confidence_intervals: list[float] | None = None,
) -> DataFrame:
    """Add summary statistics to predictions.

    Args:
        predictions: DataFrame with 'samples' column.
        confidence_intervals: CI levels (default: [0.5, 0.9, 0.95]).

    Returns:
        DataFrame with added mean, median, and CI columns.
    """
    if confidence_intervals is None:
        confidence_intervals = [0.5, 0.9, 0.95]

    if "samples" not in predictions.columns:
        raise ValueError("Predictions must have 'samples' column.")

    # Copy all existing columns
    data = {col: list(predictions[col]) for col in predictions.columns}

    # Add mean and median
    samples_column = _get_column(predictions, "samples")
    means = []
    medians = []
    for idx in range(len(predictions)):
        samples = samples_column[idx]
        means.append(float(np.mean(samples)))
        medians.append(float(np.median(samples)))

    data["mean"] = means
    data["median"] = medians

    # Add confidence intervals
    for ci in confidence_intervals:
        lower_prob = (1 - ci) / 2
        upper_prob = 1 - lower_prob
        lower_values = []
        upper_values = []

        for idx in range(len(predictions)):
            samples = samples_column[idx]
            lower_values.append(float(np.quantile(samples, lower_prob)))
            upper_values.append(float(np.quantile(samples, upper_prob)))

        data[f"ci_{ci}_lower"] = lower_values
        data[f"ci_{ci}_upper"] = upper_values

    return DataFrame.from_dict(data)
