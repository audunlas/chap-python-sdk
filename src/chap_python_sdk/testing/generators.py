"""Dataset generators for model testing based on MLServiceInfo."""

from datetime import date, timedelta
from enum import StrEnum
from typing import Callable

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from chapkit.data import DataFrame
from pydantic import BaseModel, Field

from chap_python_sdk.testing.types import ExampleData, RunInfo


class PeriodType(StrEnum):
    """Supported time period types for the model."""

    any = "any"
    week = "week"
    month = "month"
    year = "year"


class MLServiceInfo(BaseModel):
    """Model service information for dataset generation."""

    required_covariates: list[str] = Field(default_factory=list)
    allow_free_additional_continuous_covariates: bool = False
    supported_period_type: PeriodType = PeriodType.any


class DataGenerationConfig(BaseModel):
    """Configuration for test data generation."""

    prediction_length: int | None = Field(
        default=None,
        description="Number of periods to predict. Defaults based on period type.",
    )
    additional_covariates: list[str] = Field(
        default_factory=list,
        description="Additional covariates to include (if model allows).",
    )
    n_locations: int = Field(
        default=3,
        ge=1,
        description="Number of locations to generate.",
    )
    n_training_periods: int = Field(
        default=24,
        ge=1,
        description="Number of training periods.",
    )
    n_historic_periods: int = Field(
        default=12,
        ge=1,
        description="Number of historic periods (subset of training).",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility.",
    )
    location_names: list[str] | None = Field(
        default=None,
        description="Custom location names. Defaults to Location_A, Location_B, etc.",
    )
    include_nans: bool = Field(
        default=False,
        description="Include NaN values to test missing value handling.",
    )
    nan_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of values to replace with NaN when include_nans=True.",
    )


def generate_run_info(
    service_info: MLServiceInfo,
    config: DataGenerationConfig | None = None,
) -> RunInfo:
    """Generate a valid RunInfo based on MLServiceInfo.

    Args:
        service_info: Model service information with requirements.
        config: Generation configuration with prediction_length and covariates.

    Returns:
        RunInfo configured for the model.

    Raises:
        ValueError: If additional covariates requested but not allowed.
    """
    config = config or DataGenerationConfig()

    default_lengths = {
        PeriodType.week: 4,
        PeriodType.month: 3,
        PeriodType.year: 1,
        PeriodType.any: 3,
    }

    length = config.prediction_length or default_lengths.get(service_info.supported_period_type, 3)

    covariates = config.additional_covariates
    if covariates and not service_info.allow_free_additional_continuous_covariates:
        raise ValueError(
            "Model does not allow additional covariates "
            "(allow_free_additional_continuous_covariates=False)"
        )

    return RunInfo(
        prediction_length=length,
        additional_continuous_covariates=covariates,
    )


def _generate_time_periods(period_type: PeriodType, n_periods: int, start_date: date | None = None) -> list[str]:
    """Generate time period strings based on period type."""
    periods: list[str] = []
    start = start_date or date(2020, 1, 1)

    if period_type == PeriodType.month or period_type == PeriodType.any:
        for i in range(n_periods):
            year = start.year + (start.month + i - 1) // 12
            month = (start.month + i - 1) % 12 + 1
            periods.append(f"{year}-{month:02d}")

    elif period_type == PeriodType.week:
        week_start = start - timedelta(days=start.weekday())
        for i in range(n_periods):
            current_start = week_start + timedelta(weeks=i)
            current_end = current_start + timedelta(days=6)
            periods.append(f"{current_start.isoformat()}/{current_end.isoformat()}")

    elif period_type == PeriodType.year:
        for i in range(n_periods):
            periods.append(str(start.year + i))

    return periods


type CovariateGenerator = Callable[[int], np.ndarray]


def _generate_rainfall(n: int) -> np.ndarray:
    """Generate rainfall values."""
    return np.random.uniform(0, 500, n)


def _generate_temperature(n: int) -> np.ndarray:
    """Generate temperature values."""
    return np.random.uniform(15, 35, n)


def _generate_humidity(n: int) -> np.ndarray:
    """Generate humidity values."""
    return np.random.uniform(30, 100, n)


def _generate_population(n: int) -> np.ndarray:
    """Generate population values."""
    return np.random.randint(10000, 500000, n)


def _generate_default(n: int) -> np.ndarray:
    """Generate default covariate values."""
    return np.random.uniform(0, 100, n)


def _get_covariate_generator(covariate_name: str) -> CovariateGenerator:
    """Get a generator function for a specific covariate."""
    generators: dict[str, CovariateGenerator] = {
        "rainfall": _generate_rainfall,
        "mean_temperature": _generate_temperature,
        "temperature": _generate_temperature,
        "humidity": _generate_humidity,
        "population": _generate_population,
    }
    return generators.get(covariate_name, _generate_default)


def _inject_nans(df: pd.DataFrame, nan_fraction: float, exclude_columns: list[str]) -> pd.DataFrame:
    """Inject NaN values into numeric columns of a DataFrame."""
    df = df.copy()
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_columns]

    for col in numeric_columns:
        n_nans = int(len(df) * nan_fraction)
        if n_nans > 0:
            nan_indices = np.random.choice(len(df), size=n_nans, replace=False)
            df.loc[nan_indices, col] = np.nan

    return df


def generate_example_data(
    service_info: MLServiceInfo,
    run_info: RunInfo,
    config: DataGenerationConfig | None = None,
) -> ExampleData:
    """Generate complete example dataset based on service info and run info.

    Args:
        service_info: Model service information with requirements.
        run_info: Runtime information including prediction length.
        config: Generation configuration with data parameters.

    Returns:
        ExampleData with training_data, historic_data, future_data, and run_info.
    """
    config = config or DataGenerationConfig()

    if config.seed is not None:
        np.random.seed(config.seed)

    locations = config.location_names or [f"Location_{chr(65 + i)}" for i in range(config.n_locations)]

    total_periods = config.n_training_periods + run_info.prediction_length
    all_periods = _generate_time_periods(service_info.supported_period_type, total_periods)

    training_periods = all_periods[: config.n_training_periods]
    historic_start = max(0, config.n_training_periods - config.n_historic_periods)
    historic_periods = all_periods[historic_start : config.n_training_periods]
    future_periods = all_periods[config.n_training_periods :]

    all_covariates = list(service_info.required_covariates) + list(run_info.additional_continuous_covariates)

    def generate_dataframe(periods: list[str], include_target: bool) -> pd.DataFrame:
        """Generate pandas DataFrame for given periods."""
        rows: list[dict[str, object]] = []

        for period in periods:
            for location in locations:
                row: dict[str, object] = {
                    "time_period": period,
                    "location": location,
                    "population": int(_get_covariate_generator("population")(1)[0]),
                }

                if include_target:
                    row["disease_cases"] = int(np.random.poisson(50))

                for covariate in all_covariates:
                    generator = _get_covariate_generator(covariate)
                    row[covariate] = float(generator(1)[0])

                rows.append(row)

        return pd.DataFrame(rows)

    training_df = generate_dataframe(training_periods, include_target=True)
    historic_df = generate_dataframe(historic_periods, include_target=True)
    future_df = generate_dataframe(future_periods, include_target=False)

    if config.include_nans:
        exclude_cols = ["time_period", "location"]
        training_df = _inject_nans(training_df, config.nan_fraction, exclude_cols)
        historic_df = _inject_nans(historic_df, config.nan_fraction, exclude_cols)
        future_df = _inject_nans(future_df, config.nan_fraction, exclude_cols)

    return ExampleData(
        training_data=DataFrame.from_pandas(training_df),
        historic_data=DataFrame.from_pandas(historic_df),
        future_data=DataFrame.from_pandas(future_df),
        run_info=run_info,
    )


def generate_test_data(
    service_info: MLServiceInfo,
    config: DataGenerationConfig | None = None,
) -> ExampleData:
    """Generate complete test data based on model service info.

    This is a convenience function that combines generate_run_info and
    generate_example_data into a single call.

    Args:
        service_info: Model service information with requirements.
        config: Generation configuration with all parameters.

    Returns:
        ExampleData ready for model testing.
    """
    config = config or DataGenerationConfig()

    run_info = generate_run_info(service_info, config)

    return generate_example_data(service_info, run_info, config)
