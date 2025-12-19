"""Pytest fixtures for testing chapkit models."""

from typing import Any

import pytest
from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

from chap_python_sdk.testing import (
    ExampleData,
    GeoFeatureCollection,
    PredictFunction,
    RunInfo,
    TrainFunction,
    get_example_data,
)


@pytest.fixture
def laos_monthly_data() -> ExampleData:
    """Load Laos monthly example dataset."""
    return get_example_data(country="laos", frequency="monthly")


@pytest.fixture
def sample_training_data() -> DataFrame:
    """Create sample training data for testing."""
    return DataFrame.from_dict(
        {
            "time_period": ["2023-01", "2023-02", "2023-03", "2023-01", "2023-02", "2023-03"],
            "location": ["A", "A", "A", "B", "B", "B"],
            "disease_cases": [10, 20, 30, 15, 25, 35],
            "rainfall": [100.0, 150.0, 200.0, 110.0, 160.0, 210.0],
        }
    )


@pytest.fixture
def sample_future_data() -> DataFrame:
    """Create sample future data for testing."""
    return DataFrame.from_dict(
        {
            "time_period": ["2023-04", "2023-05", "2023-04", "2023-05"],
            "location": ["A", "A", "B", "B"],
            "rainfall": [180.0, 120.0, 190.0, 130.0],
        }
    )


@pytest.fixture
def sample_historic_data() -> DataFrame:
    """Create sample historic data for testing."""
    return DataFrame.from_dict(
        {
            "time_period": ["2023-01", "2023-02", "2023-03", "2023-01", "2023-02", "2023-03"],
            "location": ["A", "A", "A", "B", "B", "B"],
            "disease_cases": [10, 20, 30, 15, 25, 35],
            "rainfall": [100.0, 150.0, 200.0, 110.0, 160.0, 210.0],
        }
    )


@pytest.fixture
def valid_nested_predictions() -> DataFrame:
    """Create valid predictions in nested format."""
    return DataFrame.from_dict(
        {
            "time_period": ["2023-04", "2023-05", "2023-04", "2023-05"],
            "location": ["A", "A", "B", "B"],
            "samples": [[10.0, 12.0, 11.0], [15.0, 17.0, 16.0], [20.0, 22.0, 21.0], [25.0, 27.0, 26.0]],
        }
    )


@pytest.fixture
def valid_wide_predictions() -> DataFrame:
    """Create valid predictions in wide format."""
    return DataFrame.from_dict(
        {
            "time_period": ["2023-04", "2023-05", "2023-04", "2023-05"],
            "location": ["A", "A", "B", "B"],
            "sample_0": [10.0, 15.0, 20.0, 25.0],
            "sample_1": [12.0, 17.0, 22.0, 27.0],
            "sample_2": [11.0, 16.0, 21.0, 26.0],
        }
    )


@pytest.fixture
def valid_long_predictions() -> DataFrame:
    """Create valid predictions in long format."""
    return DataFrame.from_dict(
        {
            "time_period": ["2023-04", "2023-04", "2023-04", "2023-05", "2023-05", "2023-05"],
            "location": ["A", "A", "A", "A", "A", "A"],
            "sample_id": [0, 1, 2, 0, 1, 2],
            "prediction": [10.0, 12.0, 11.0, 15.0, 17.0, 16.0],
        }
    )


# Functional train/predict functions for testing


def create_simple_train_function(n_samples: int = 10) -> TrainFunction:
    """Create a simple train function that computes a mean model."""

    async def simple_train(
        config: BaseConfig,
        data: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> dict[str, Any]:
        """Train a simple mean model."""
        if "disease_cases" in data.columns:
            values = [float(v) for v in data["disease_cases"] if v != "" and v is not None]
            mean_value = sum(values) / len(values) if values else 0.0
        else:
            mean_value = 10.0
        return {"mean": mean_value, "n_samples": n_samples, "prediction_length": run_info.prediction_length}

    return simple_train


def create_simple_predict_function() -> PredictFunction:
    """Create a simple predict function that generates samples."""

    async def simple_predict(
        config: BaseConfig,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> DataFrame:
        """Generate predictions with configurable sample count."""
        mean_value = model["mean"]
        n_samples = model.get("n_samples", 10)
        samples_list = []
        for _ in range(len(future)):
            samples = [mean_value + i * 0.1 for i in range(n_samples)]
            samples_list.append(samples)

        return DataFrame.from_dict(
            {
                "time_period": list(future["time_period"]),
                "location": list(future["location"]),
                "samples": samples_list,
            }
        )

    return simple_predict


def create_failing_train_function() -> TrainFunction:
    """Create a train function that fails."""

    async def failing_train(
        config: BaseConfig,
        data: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> Any:
        """Fail during training."""
        raise ValueError("Training failed intentionally")

    return failing_train


def create_failing_predict_function() -> PredictFunction:
    """Create a predict function that fails."""

    async def failing_predict(
        config: BaseConfig,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> DataFrame:
        """Fail during prediction."""
        raise ValueError("Prediction failed intentionally")

    return failing_predict


def create_invalid_output_predict_function() -> PredictFunction:
    """Create a predict function that returns invalid output (missing samples)."""

    async def invalid_output_predict(
        config: BaseConfig,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> DataFrame:
        """Return predictions missing samples column."""
        return DataFrame.from_dict(
            {
                "time_period": list(future["time_period"]),
                "location": list(future["location"]),
            }
        )

    return invalid_output_predict


def create_wide_format_predict_function(n_samples: int = 10) -> PredictFunction:
    """Create a predict function that returns wide format predictions."""

    async def wide_format_predict(
        config: BaseConfig,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> DataFrame:
        """Generate predictions in wide format (sample_0, sample_1, etc.)."""
        mean_value = model["mean"]

        data: dict[str, Any] = {
            "time_period": list(future["time_period"]),
            "location": list(future["location"]),
        }

        for sample_idx in range(n_samples):
            col_name = f"sample_{sample_idx}"
            data[col_name] = [mean_value + sample_idx * 0.1 for _ in range(len(future))]

        return DataFrame.from_dict(data)

    return wide_format_predict


# Fixtures providing train/predict function pairs


@pytest.fixture
def simple_train_function() -> TrainFunction:
    """Create a simple train function for testing."""
    return create_simple_train_function(n_samples=10)


@pytest.fixture
def simple_predict_function() -> PredictFunction:
    """Create a simple predict function for testing."""
    return create_simple_predict_function()


@pytest.fixture
def failing_train_function() -> TrainFunction:
    """Create a train function that fails during training."""
    return create_failing_train_function()


@pytest.fixture
def failing_predict_function() -> PredictFunction:
    """Create a predict function that fails during prediction."""
    return create_failing_predict_function()


@pytest.fixture
def invalid_output_predict_function() -> PredictFunction:
    """Create a predict function that returns invalid output."""
    return create_invalid_output_predict_function()


@pytest.fixture
def wide_format_predict_function() -> PredictFunction:
    """Create a predict function that returns wide format predictions."""
    return create_wide_format_predict_function(n_samples=10)
