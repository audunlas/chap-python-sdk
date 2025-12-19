"""Tests for config validation and propagation.

These tests are designed to catch common bugs discovered during chapkit integration:
- Type mismatches between config and model internals
- Hardcoded defaults that ignore config values
- Config values not propagating through the model pipeline
"""

from typing import Any

import pytest
from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame
from pydantic import Field

from chap_python_sdk.testing import (
    ExampleData,
    GeoFeatureCollection,
    PredictFunction,
    RunInfo,
    TrainFunction,
    get_example_data,
    validate_model_io,
)
from chap_python_sdk.testing.types import ValidationResult


class ConfigWithCustomValues(BaseConfig):
    """Config with non-default values to test propagation."""

    lag: int = Field(default=3, description="Number of lag periods")
    n_samples: int = Field(default=100, description="Number of prediction samples")


def create_config_aware_train_function() -> TrainFunction:
    """Create a train function that stores config for verification."""

    async def config_aware_train(
        config: BaseConfig,
        data: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> dict[str, Any]:
        """Train and store the config for verification."""
        # Cast to our custom config type to access custom fields
        custom_config = config if isinstance(config, ConfigWithCustomValues) else ConfigWithCustomValues()
        return {
            "mean": 10.0,
            "config_lag": custom_config.lag,
            "config_n_samples": custom_config.n_samples,
            "original_config": config,
        }

    return config_aware_train


def create_config_aware_predict_function() -> PredictFunction:
    """Create a predict function that uses config values."""

    async def config_aware_predict(
        config: BaseConfig,
        model: dict[str, Any],
        historic: DataFrame,
        future: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> DataFrame:
        """Generate predictions using config values."""
        # Cast to our custom config type to access custom fields
        custom_config = config if isinstance(config, ConfigWithCustomValues) else ConfigWithCustomValues()

        # Use n_samples from config to generate samples
        n_samples = custom_config.n_samples
        samples_list = []
        for _ in range(len(future)):
            samples = [model["mean"] + i * 0.1 for i in range(n_samples)]
            samples_list.append(samples)

        return DataFrame.from_dict(
            {
                "time_period": list(future["time_period"]),
                "location": list(future["location"]),
                "samples": samples_list,
            }
        )

    return config_aware_predict


def create_hardcoded_defaults_train_function() -> TrainFunction:
    """Create a train function for testing hardcoded defaults."""

    async def hardcoded_train(
        config: BaseConfig,
        data: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> dict[str, float]:
        """Train the model."""
        return {"mean": 10.0}

    return hardcoded_train


def create_hardcoded_defaults_predict_function(hardcoded_n_samples: int = 10) -> PredictFunction:
    """Create a predict function that ignores config values (simulates bug)."""

    async def hardcoded_predict(
        config: BaseConfig,
        model: dict[str, float],
        historic: DataFrame,
        future: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> DataFrame:
        """Generate predictions with hardcoded sample count (ignoring config)."""
        # BUG: Uses hardcoded value instead of config.n_samples
        n_samples = hardcoded_n_samples
        samples_list = []
        for _ in range(len(future)):
            samples = [model["mean"] + i * 0.1 for i in range(n_samples)]
            samples_list.append(samples)

        return DataFrame.from_dict(
            {
                "time_period": list(future["time_period"]),
                "location": list(future["location"]),
                "samples": samples_list,
            }
        )

    return hardcoded_predict


@pytest.fixture
def laos_data() -> ExampleData:
    """Load Laos monthly example dataset."""
    return get_example_data(country="laos", frequency="monthly")


@pytest.fixture
def config_aware_train() -> TrainFunction:
    """Create a config-aware train function."""
    return create_config_aware_train_function()


@pytest.fixture
def config_aware_predict() -> PredictFunction:
    """Create a config-aware predict function."""
    return create_config_aware_predict_function()


class TestConfigPropagation:
    """Tests for verifying config values propagate through the model."""

    @pytest.mark.asyncio
    async def test_default_config_works(
        self,
        config_aware_train: TrainFunction,
        config_aware_predict: PredictFunction,
        laos_data: ExampleData,
    ) -> None:
        """Test that model works with default config values."""
        config = ConfigWithCustomValues()

        result = await validate_model_io(config_aware_train, config_aware_predict, laos_data, config)

        assert result.success, f"Validation failed: {result.errors}"

    @pytest.mark.asyncio
    async def test_custom_config_values_propagate(
        self,
        config_aware_train: TrainFunction,
        config_aware_predict: PredictFunction,
        laos_data: ExampleData,
    ) -> None:
        """Test that custom config values are actually used by the model."""
        config = ConfigWithCustomValues(lag=5, n_samples=50)

        result = await validate_model_io(config_aware_train, config_aware_predict, laos_data, config)

        assert result.success, f"Validation failed: {result.errors}"
        # Verify the sample count matches what we configured
        assert result.n_samples == 50

    @pytest.mark.asyncio
    async def test_sample_count_matches_config(
        self,
        config_aware_train: TrainFunction,
        config_aware_predict: PredictFunction,
        laos_data: ExampleData,
    ) -> None:
        """Test that n_samples config actually affects prediction output."""
        expected_samples = 25
        config = ConfigWithCustomValues(n_samples=expected_samples)

        result = await validate_model_io(config_aware_train, config_aware_predict, laos_data, config)

        assert result.success, f"Validation failed: {result.errors}"
        assert result.n_samples == expected_samples, (
            f"Expected {expected_samples} samples from config, got {result.n_samples}"
        )


class TestHardcodedDefaultsDetection:
    """Tests to detect models that ignore config values."""

    @pytest.mark.asyncio
    async def test_detects_hardcoded_sample_count(self, laos_data: ExampleData) -> None:
        """Test that we can detect when a model ignores n_samples config.

        This test demonstrates that validation correctly reports the actual
        sample count, which allows users to detect when their model has
        hardcoded defaults instead of using config values.
        """
        train_function = create_hardcoded_defaults_train_function()
        predict_function = create_hardcoded_defaults_predict_function(hardcoded_n_samples=10)
        expected_samples = 25  # Different from hardcoded value (10)
        config = ConfigWithCustomValues(n_samples=expected_samples)

        result = await validate_model_io(train_function, predict_function, laos_data, config)

        # The validation passes (model produces valid output)
        assert result.success, f"Validation failed: {result.errors}"

        # The validation correctly reports the actual sample count (10),
        # which differs from expected (25), allowing detection of hardcoded defaults
        assert result.n_samples == 10, "Should report actual sample count from hardcoded model"
        assert result.n_samples != expected_samples, "Hardcoded model should ignore config"


class TestConfigTypeValidation:
    """Tests for config type compatibility."""

    @pytest.mark.asyncio
    async def test_config_with_valid_types(
        self,
        config_aware_train: TrainFunction,
        config_aware_predict: PredictFunction,
        laos_data: ExampleData,
    ) -> None:
        """Test that config with valid types works correctly."""
        config = ConfigWithCustomValues(lag=3, n_samples=100)

        result = await validate_model_io(config_aware_train, config_aware_predict, laos_data, config)

        assert result.success, f"Validation failed: {result.errors}"

    def test_config_rejects_invalid_types(self) -> None:
        """Test that pydantic rejects invalid config types."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ConfigWithCustomValues(lag="not_an_int")  # type: ignore[arg-type]

        with pytest.raises(Exception):  # Pydantic ValidationError
            ConfigWithCustomValues(n_samples=1.5)  # type: ignore[arg-type]


def assert_config_affects_output(
    result_with_default: ValidationResult,
    result_with_custom: ValidationResult,
    config_field: str,
) -> None:
    """Assert that changing a config field affects model output.

    This helper can be used to verify config propagation.
    """
    # If outputs are identical despite different configs, something is wrong
    if result_with_default.n_samples == result_with_custom.n_samples:
        if config_field == "n_samples":
            raise AssertionError(
                f"Config field '{config_field}' had no effect on output. "
                f"Both configs produced {result_with_default.n_samples} samples."
            )
