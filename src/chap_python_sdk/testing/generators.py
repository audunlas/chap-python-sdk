"""Dataset generators for model testing based on MLServiceInfo."""

from datetime import date, timedelta
from enum import StrEnum
from typing import Callable

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from chapkit.data import DataFrame

from chap_python_sdk.testing.types import ExampleData, RunInfo


class PeriodType(StrEnum):
    """Supported time period types for the model."""

    any = "any"
    week = "week"
    month = "month"
    year = "year"


class MLServiceInfo:
    """Minimal representation of MLServiceInfo for dataset generation.

    This allows using the generator without importing chapkit.api directly,
    or accepting the actual MLServiceInfo from chapkit.
    """

    def __init__(
        self,
        required_covariates: list[str] | None = None,
        allow_free_additional_continuous_covariates: bool = False,
        supported_period_type: PeriodType | str = PeriodType.any,
    ) -> None:
        """Initialize service info for dataset generation."""
        self.required_covariates = required_covariates or []
        self.allow_free_additional_continuous_covariates = allow_free_additional_continuous_covariates
        self.supported_period_type = (
            PeriodType(supported_period_type)
            if isinstance(supported_period_type, str)
            else supported_period_type
        )


def generate_run_info(
    service_info: MLServiceInfo,
    prediction_length: int | None = None,
    additional_covariates: list[str] | None = None,
) -> RunInfo:
    """Generate a valid RunInfo based on MLServiceInfo.

    Args:
        service_info: Model service information with requirements.
        prediction_length: Override default prediction length.
        additional_covariates: Additional covariates to include (if allowed).

    Returns:
        RunInfo configured for the model.

    Raises:
        ValueError: If additional covariates requested but not allowed.
    """
    default_lengths = {
        PeriodType.week: 4,
        PeriodType.month: 3,
        PeriodType.year: 1,
        PeriodType.any: 3,
    }

    length = prediction_length or default_lengths.get(service_info.supported_period_type, 3)

    covariates = additional_covariates or []
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


def generate_example_data(
    service_info: MLServiceInfo,
    run_info: RunInfo,
    n_locations: int = 3,
    n_training_periods: int = 24,
    n_historic_periods: int = 12,
    seed: int | None = None,
    location_names: list[str] | None = None,
) -> ExampleData:
    """Generate complete example dataset based on service info and run info.

    Args:
        service_info: Model service information with requirements.
        run_info: Runtime information including prediction length.
        n_locations: Number of locations to generate.
        n_training_periods: Number of training periods.
        n_historic_periods: Number of historic periods (subset of training).
        seed: Random seed for reproducibility.
        location_names: Custom location names (default: Location_A, Location_B, ...).

    Returns:
        ExampleData with training_data, historic_data, future_data, and run_info.
    """
    if seed is not None:
        np.random.seed(seed)

    locations = location_names or [f"Location_{chr(65 + i)}" for i in range(n_locations)]

    total_periods = n_training_periods + run_info.prediction_length
    all_periods = _generate_time_periods(service_info.supported_period_type, total_periods)

    training_periods = all_periods[:n_training_periods]
    historic_start = max(0, n_training_periods - n_historic_periods)
    historic_periods = all_periods[historic_start:n_training_periods]
    future_periods = all_periods[n_training_periods:]

    all_covariates = list(service_info.required_covariates) + list(run_info.additional_continuous_covariates)

    def generate_dataframe(periods: list[str], include_target: bool) -> DataFrame:
        """Generate DataFrame for given periods."""
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

        return DataFrame.from_pandas(pd.DataFrame(rows))

    training_data = generate_dataframe(training_periods, include_target=True)
    historic_data = generate_dataframe(historic_periods, include_target=True)
    future_data = generate_dataframe(future_periods, include_target=False)

    return ExampleData(
        training_data=training_data,
        historic_data=historic_data,
        future_data=future_data,
        run_info=run_info,
    )


def generate_test_data(
    service_info: MLServiceInfo,
    prediction_length: int | None = None,
    additional_covariates: list[str] | None = None,
    n_locations: int = 3,
    n_training_periods: int = 24,
    n_historic_periods: int = 12,
    seed: int | None = None,
    location_names: list[str] | None = None,
) -> ExampleData:
    """Generate complete test data based on model service info.

    This is a convenience function that combines generate_run_info and
    generate_example_data into a single call.

    Args:
        service_info: Model service information with requirements.
        prediction_length: Override default prediction length.
        additional_covariates: Additional covariates to include (if allowed).
        n_locations: Number of locations to generate.
        n_training_periods: Number of training periods.
        n_historic_periods: Number of historic periods.
        seed: Random seed for reproducibility.
        location_names: Custom location names.

    Returns:
        ExampleData ready for model testing.
    """
    run_info = generate_run_info(
        service_info,
        prediction_length=prediction_length,
        additional_covariates=additional_covariates,
    )

    return generate_example_data(
        service_info,
        run_info,
        n_locations=n_locations,
        n_training_periods=n_training_periods,
        n_historic_periods=n_historic_periods,
        seed=seed,
        location_names=location_names,
    )
