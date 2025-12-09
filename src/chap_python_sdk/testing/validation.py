"""Model I/O validation for chapkit models."""

import traceback
from typing import Any

from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

from chap_python_sdk.testing.assertions import PredictionValidationError
from chap_python_sdk.testing.example_data import get_example_data, list_available_datasets
from chap_python_sdk.testing.predictions import detect_prediction_format, predictions_from_wide
from chap_python_sdk.testing.types import ExampleData, ModelRunnerProtocol, ValidationResult


async def validate_model_io(
    runner: ModelRunnerProtocol[Any],
    example_data: ExampleData,
    config: BaseConfig | None = None,
) -> ValidationResult:
    """Validate model runner against example data.

    This function tests that a model runner correctly implements the
    BaseModelRunner interface by running it against example data and
    validating the output.

    Args:
        runner: Model runner implementing ModelRunnerProtocol.
        example_data: Example dataset to test against.
        config: Optional model configuration.

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

    try:
        # Step 1: Call on_init
        await runner.on_init()
    except Exception as exception:
        errors.append(f"on_init() failed: {exception}")
        return ValidationResult(
            success=False,
            errors=errors,
            warnings=warnings,
            n_predictions=0,
            n_samples=0,
        )

    trained_model: Any = None
    try:
        # Step 2: Train the model
        trained_model = await runner.on_train(
            config=config,
            data=example_data.training_data,
            geo=example_data.geo,
        )
    except Exception as exception:
        errors.append(f"on_train() failed: {exception}\n{traceback.format_exc()}")

    predictions: DataFrame | None = None
    if trained_model is not None:
        try:
            # Step 3: Generate predictions
            predictions = await runner.on_predict(
                config=config,
                model=trained_model,
                historic=example_data.historic_data,
                future=example_data.future_data,
                geo=example_data.geo,
            )
        except Exception as exception:
            errors.append(f"on_predict() failed: {exception}\n{traceback.format_exc()}")

    # Step 4: Validate predictions
    if predictions is not None:
        validation_errors = _validate_predictions(predictions, example_data.future_data)
        errors.extend(validation_errors)

        if not validation_errors:
            n_predictions = len(predictions)
            if "samples" in predictions.columns and n_predictions > 0:
                n_samples = len(predictions["samples"].iloc[0])

    try:
        # Step 5: Call on_cleanup
        await runner.on_cleanup()
    except Exception as exception:
        warnings.append(f"on_cleanup() failed: {exception}")

    return ValidationResult(
        success=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        n_predictions=n_predictions,
        n_samples=n_samples,
    )


def _validate_predictions(predictions: DataFrame, future_data: DataFrame) -> list[str]:
    """Validate prediction output structure.

    Args:
        predictions: Predictions DataFrame.
        future_data: Expected future data DataFrame.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    # Check if predictions is a DataFrame
    if not isinstance(predictions, DataFrame):
        errors.append(f"Predictions must be a DataFrame, got {type(predictions).__name__}")
        return errors

    # Check if predictions is empty
    if len(predictions) == 0:
        errors.append("Predictions DataFrame is empty")
        return errors

    # Try to detect and convert format if needed
    try:
        prediction_format = detect_prediction_format(predictions)
        if prediction_format == "wide":
            predictions = predictions_from_wide(predictions)
    except ValueError as exception:
        errors.append(f"Invalid prediction format: {exception}")
        return errors

    # Check required columns
    if "time_period" not in predictions.columns:
        errors.append("Predictions missing 'time_period' column")

    if "location" not in predictions.columns:
        errors.append("Predictions missing 'location' column")

    if "samples" not in predictions.columns:
        errors.append("Predictions missing 'samples' column")
        return errors

    # Check row count
    if len(predictions) != len(future_data):
        errors.append(
            f"Predictions have {len(predictions)} rows, "
            f"expected {len(future_data)} (matching future_data)"
        )

    # Check samples column
    try:
        _validate_samples_column(predictions)
    except PredictionValidationError as exception:
        errors.append(str(exception))

    return errors


def _validate_samples_column(predictions: DataFrame) -> None:
    """Validate samples column structure.

    Args:
        predictions: DataFrame with samples column.

    Raises:
        PredictionValidationError: If validation fails.
    """
    sample_counts: set[int] = set()

    for idx in range(len(predictions)):
        samples = predictions["samples"].iloc[idx]

        if not isinstance(samples, (list,)):
            raise PredictionValidationError(
                f"Row {idx}: 'samples' must be a list, got {type(samples).__name__}"
            )

        if len(samples) == 0:
            raise PredictionValidationError(f"Row {idx}: 'samples' is empty")

        sample_counts.add(len(samples))

        # Check that all samples are numeric
        for sample_idx, value in enumerate(samples):
            if not isinstance(value, (int, float)):
                raise PredictionValidationError(
                    f"Row {idx}, sample {sample_idx}: "
                    f"Expected numeric value, got {type(value).__name__}"
                )

    # Check consistent sample counts
    if len(sample_counts) > 1:
        raise PredictionValidationError(
            f"Inconsistent sample counts across rows: {sample_counts}"
        )


async def validate_model_io_all(
    runner: ModelRunnerProtocol[Any],
    config: BaseConfig | None = None,
    country: str | None = None,
    frequency: str | None = None,
) -> ValidationResult:
    """Validate model runner against all matching datasets.

    Args:
        runner: Model runner implementing ModelRunnerProtocol.
        config: Optional model configuration.
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
        result = await validate_model_io(runner, example_data, config)

        if result.errors:
            all_errors.extend(
                [f"[{dataset_country}/{dataset_frequency}] {e}" for e in result.errors]
            )

        if result.warnings:
            all_warnings.extend(
                [f"[{dataset_country}/{dataset_frequency}] {w}" for w in result.warnings]
            )

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
