"""Model I/O validation for chapkit models."""

import traceback
from typing import Any, cast

from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

from chap_python_sdk.testing.assertions import PredictionValidationError
from chap_python_sdk.testing.example_data import get_example_data, list_available_datasets
from chap_python_sdk.testing.predictions import detect_prediction_format, predictions_from_wide
from chap_python_sdk.testing.types import (
    ExampleData,
    PredictFunction,
    RunInfo,
    TrainFunction,
    ValidationResult,
)


def _get_column(dataframe: DataFrame, column: str) -> list[Any]:
    """Get a column from a DataFrame as a list."""
    return cast(list[Any], dataframe[column])


async def validate_model_io(
    train_function: TrainFunction,
    predict_function: PredictFunction,
    example_data: ExampleData,
    config: BaseConfig | None = None,
    run_info: RunInfo | None = None,
) -> ValidationResult:
    """Validate model train/predict functions against example data.

    This function tests that train and predict functions correctly implement
    the chapkit functional interface by running them against example data and
    validating the output.

    Args:
        train_function: Async function that trains a model.
        predict_function: Async function that generates predictions.
        example_data: Example dataset to test against.
        config: Optional model configuration.
        run_info: Optional runtime information. If not provided, uses example_data.run_info
            or creates a default RunInfo.

    Returns:
        ValidationResult with success status, errors, and statistics.
    """
    errors: list[str] = []
    warnings: list[str] = []
    n_predictions = 0
    n_samples = 0

    # Use provided config or create a default one
    if config is None:
        config = BaseConfig()

    # Use provided run_info, or from example_data, or create default
    if run_info is None:
        run_info = example_data.run_info
    if run_info is None:
        run_info = RunInfo(prediction_length=len(example_data.future_data))

    trained_model: Any = None
    try:
        # Step 1: Train the model
        trained_model = await train_function(
            config,
            example_data.training_data,
            run_info,
            example_data.geo,
        )
    except Exception as exception:
        errors.append(f"train_function() failed: {exception}\n{traceback.format_exc()}")

    predictions: DataFrame | None = None
    if trained_model is not None:
        try:
            # Step 2: Generate predictions
            predictions = await predict_function(
                config,
                trained_model,
                example_data.historic_data,
                example_data.future_data,
                run_info,
                example_data.geo,
            )
        except Exception as exception:
            errors.append(f"predict_function() failed: {exception}\n{traceback.format_exc()}")

    # Step 3: Validate predictions
    if predictions is not None:
        normalized_predictions, validation_errors = _validate_predictions(predictions, example_data.future_data)
        errors.extend(validation_errors)

        if not validation_errors and normalized_predictions is not None:
            n_predictions = len(normalized_predictions)
            if "samples" in normalized_predictions.columns and n_predictions > 0:
                n_samples = len(_get_column(normalized_predictions, "samples")[0])

    return ValidationResult(
        success=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        n_predictions=n_predictions,
        n_samples=n_samples,
    )


def _validate_predictions(predictions: DataFrame, future_data: DataFrame) -> tuple[DataFrame | None, list[str]]:
    """Validate prediction output structure.

    Args:
        predictions: Predictions DataFrame.
        future_data: Expected future data DataFrame.

    Returns:
        Tuple of (normalized_predictions, error_messages).
        normalized_predictions is None if validation fails early.
    """
    errors: list[str] = []

    # Check if predictions is a DataFrame
    if not isinstance(predictions, DataFrame):
        errors.append(f"Predictions must be a DataFrame, got {type(predictions).__name__}")
        return None, errors

    # Check if predictions is empty
    if len(predictions) == 0:
        errors.append("Predictions DataFrame is empty")
        return None, errors

    # Try to detect and convert format if needed
    try:
        prediction_format = detect_prediction_format(predictions)
        if prediction_format == "wide":
            predictions = predictions_from_wide(predictions)
    except ValueError as exception:
        errors.append(f"Invalid prediction format: {exception}")
        return None, errors

    # Check required columns
    if "time_period" not in predictions.columns:
        errors.append("Predictions missing 'time_period' column")

    if "location" not in predictions.columns:
        errors.append("Predictions missing 'location' column")

    if "samples" not in predictions.columns:
        errors.append("Predictions missing 'samples' column")
        return None, errors

    # Check row count
    if len(predictions) != len(future_data):
        errors.append(f"Predictions have {len(predictions)} rows, expected {len(future_data)} (matching future_data)")

    # Check samples column
    try:
        _validate_samples_column(predictions)
    except PredictionValidationError as exception:
        errors.append(str(exception))

    return predictions, errors


def _validate_samples_column(predictions: DataFrame) -> None:
    """Validate samples column structure.

    Args:
        predictions: DataFrame with samples column.

    Raises:
        PredictionValidationError: If validation fails.
    """
    sample_counts: set[int] = set()
    samples_column = _get_column(predictions, "samples")

    for idx in range(len(predictions)):
        samples = samples_column[idx]

        if not isinstance(samples, (list,)):
            raise PredictionValidationError(f"Row {idx}: 'samples' must be a list, got {type(samples).__name__}")

        if len(samples) == 0:
            raise PredictionValidationError(f"Row {idx}: 'samples' is empty")

        sample_counts.add(len(samples))

        # Check that all samples are numeric
        for sample_idx, value in enumerate(samples):
            if not isinstance(value, (int, float)):
                raise PredictionValidationError(
                    f"Row {idx}, sample {sample_idx}: Expected numeric value, got {type(value).__name__}"
                )

    # Check consistent sample counts
    if len(sample_counts) > 1:
        raise PredictionValidationError(f"Inconsistent sample counts across rows: {sample_counts}")


async def validate_model_io_all(
    train_function: TrainFunction,
    predict_function: PredictFunction,
    config: BaseConfig | None = None,
    run_info: RunInfo | None = None,
    country: str | None = None,
    frequency: str | None = None,
) -> ValidationResult:
    """Validate model train/predict functions against all matching datasets.

    Args:
        train_function: Async function that trains a model.
        predict_function: Async function that generates predictions.
        config: Optional model configuration.
        run_info: Optional runtime information. If not provided, uses dataset defaults.
        country: Filter by country (optional).
        frequency: Filter by frequency (optional).

    Returns:
        Combined ValidationResult from all dataset validations.
    """
    datasets = list_available_datasets()

    # Filter datasets
    if country is not None:
        datasets = [(c, f) for c, f in datasets if c.lower() == country.lower()]

    if frequency is not None:
        datasets = [(c, f) for c, f in datasets if f.lower() == frequency.lower()]

    if not datasets:
        return ValidationResult(
            success=False,
            errors=["No matching datasets found"],
            warnings=[],
            n_predictions=0,
            n_samples=0,
        )

    all_errors: list[str] = []
    all_warnings: list[str] = []
    total_predictions = 0
    n_samples = 0

    for dataset_country, dataset_frequency in datasets:
        example_data = get_example_data(dataset_country, dataset_frequency)
        result = await validate_model_io(train_function, predict_function, example_data, config, run_info)

        if result.errors:
            all_errors.extend([f"[{dataset_country}/{dataset_frequency}] {e}" for e in result.errors])

        if result.warnings:
            all_warnings.extend([f"[{dataset_country}/{dataset_frequency}] {w}" for w in result.warnings])

        total_predictions += result.n_predictions
        if result.n_samples > 0:
            n_samples = result.n_samples

    return ValidationResult(
        success=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings,
        n_predictions=total_predictions,
        n_samples=n_samples,
    )
