"""Tests for model I/O validation."""

from chapkit.config.schemas import BaseConfig

from chap_python_sdk.testing import (
    ExampleData,
    PredictFunction,
    TrainFunction,
    validate_model_io,
    validate_model_io_all,
)

from .conftest import (
    create_simple_predict_function,
    create_simple_train_function,
    create_wide_format_predict_function,
)


class TestValidateModelIO:
    """Tests for validate_model_io function."""

    async def test_validates_simple_model(
        self,
        simple_train_function: TrainFunction,
        simple_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test validation with a simple deterministic model."""
        result = await validate_model_io(simple_train_function, simple_predict_function, laos_monthly_data)
        assert result.success is True
        assert len(result.errors) == 0

    async def test_returns_prediction_count(
        self,
        simple_train_function: TrainFunction,
        simple_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that validation returns correct prediction count."""
        result = await validate_model_io(simple_train_function, simple_predict_function, laos_monthly_data)
        assert result.n_predictions > 0
        assert result.n_predictions == len(laos_monthly_data.future_data)

    async def test_returns_sample_count(
        self,
        simple_train_function: TrainFunction,
        simple_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that validation returns correct sample count."""
        result = await validate_model_io(simple_train_function, simple_predict_function, laos_monthly_data)
        assert result.n_samples == 10

    async def test_fails_when_train_raises(
        self,
        failing_train_function: TrainFunction,
        simple_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that validation fails when train function raises."""
        result = await validate_model_io(failing_train_function, simple_predict_function, laos_monthly_data)
        assert result.success is False
        assert any("train_function() failed" in error for error in result.errors)

    async def test_fails_when_predict_raises(
        self,
        simple_train_function: TrainFunction,
        failing_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that validation fails when predict function raises."""
        result = await validate_model_io(simple_train_function, failing_predict_function, laos_monthly_data)
        assert result.success is False
        assert any("predict_function() failed" in error for error in result.errors)

    async def test_fails_for_invalid_output(
        self,
        simple_train_function: TrainFunction,
        invalid_output_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that validation fails for invalid prediction output."""
        result = await validate_model_io(simple_train_function, invalid_output_predict_function, laos_monthly_data)
        assert result.success is False
        assert any("samples" in error.lower() for error in result.errors)

    async def test_uses_default_config(
        self,
        simple_train_function: TrainFunction,
        simple_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that validation works with default config."""
        result = await validate_model_io(simple_train_function, simple_predict_function, laos_monthly_data)
        assert result.success is True

    async def test_accepts_custom_config(
        self,
        simple_train_function: TrainFunction,
        simple_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that validation accepts custom config."""
        config = BaseConfig()
        result = await validate_model_io(
            simple_train_function, simple_predict_function, laos_monthly_data, config=config
        )
        assert result.success is True


class TestValidateModelIOWithDifferentSampleCounts:
    """Tests for validate_model_io with different sample counts."""

    async def test_single_sample(self, laos_monthly_data: ExampleData) -> None:
        """Test validation with single sample (deterministic model)."""
        train_function = create_simple_train_function(n_samples=1)
        predict_function = create_simple_predict_function()
        result = await validate_model_io(train_function, predict_function, laos_monthly_data)
        assert result.success is True
        assert result.n_samples == 1

    async def test_many_samples(self, laos_monthly_data: ExampleData) -> None:
        """Test validation with many samples (probabilistic model)."""
        train_function = create_simple_train_function(n_samples=100)
        predict_function = create_simple_predict_function()
        result = await validate_model_io(train_function, predict_function, laos_monthly_data)
        assert result.success is True
        assert result.n_samples == 100


class TestValidateModelIOWideFormat:
    """Tests for validate_model_io with wide format predictions."""

    async def test_validates_wide_format_model(
        self,
        simple_train_function: TrainFunction,
        wide_format_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test validation with model that returns wide format predictions."""
        result = await validate_model_io(simple_train_function, wide_format_predict_function, laos_monthly_data)
        assert result.success is True
        assert len(result.errors) == 0

    async def test_counts_samples_correctly_for_wide_format(
        self,
        simple_train_function: TrainFunction,
        wide_format_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that n_samples is counted correctly for wide format predictions."""
        result = await validate_model_io(simple_train_function, wide_format_predict_function, laos_monthly_data)
        assert result.success is True
        assert result.n_samples == 10, f"Expected 10 samples, got {result.n_samples}"

    async def test_wide_format_with_different_sample_counts(self, laos_monthly_data: ExampleData) -> None:
        """Test wide format validation with various sample counts."""
        for expected_samples in [1, 5, 25, 100]:
            train_function = create_simple_train_function(n_samples=expected_samples)
            predict_function = create_wide_format_predict_function(n_samples=expected_samples)
            result = await validate_model_io(train_function, predict_function, laos_monthly_data)
            assert result.success is True
            assert result.n_samples == expected_samples, f"Expected {expected_samples} samples, got {result.n_samples}"


class TestValidateModelIOAll:
    """Tests for validate_model_io_all function."""

    async def test_validates_all_datasets(
        self, simple_train_function: TrainFunction, simple_predict_function: PredictFunction
    ) -> None:
        """Test validation against all datasets."""
        result = await validate_model_io_all(simple_train_function, simple_predict_function)
        assert result.success is True

    async def test_filters_by_country(
        self, simple_train_function: TrainFunction, simple_predict_function: PredictFunction
    ) -> None:
        """Test validation with country filter."""
        result = await validate_model_io_all(simple_train_function, simple_predict_function, country="laos")
        assert result.success is True

    async def test_filters_by_frequency(
        self, simple_train_function: TrainFunction, simple_predict_function: PredictFunction
    ) -> None:
        """Test validation with frequency filter."""
        result = await validate_model_io_all(simple_train_function, simple_predict_function, frequency="monthly")
        assert result.success is True

    async def test_returns_total_predictions(
        self, simple_train_function: TrainFunction, simple_predict_function: PredictFunction
    ) -> None:
        """Test that total predictions are returned."""
        result = await validate_model_io_all(simple_train_function, simple_predict_function)
        assert result.n_predictions > 0

    async def test_fails_for_no_matching_datasets(
        self, simple_train_function: TrainFunction, simple_predict_function: PredictFunction
    ) -> None:
        """Test that validation fails when no datasets match filter."""
        result = await validate_model_io_all(simple_train_function, simple_predict_function, country="nonexistent")
        assert result.success is False
        assert any("No matching datasets" in error for error in result.errors)

    async def test_collects_errors_from_all_datasets(
        self, failing_train_function: TrainFunction, simple_predict_function: PredictFunction
    ) -> None:
        """Test that errors from all datasets are collected."""
        result = await validate_model_io_all(failing_train_function, simple_predict_function)
        assert result.success is False
        assert len(result.errors) >= 1


class TestValidationResultFields:
    """Tests for ValidationResult field values."""

    async def test_success_is_boolean(
        self,
        simple_train_function: TrainFunction,
        simple_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that success is a boolean."""
        result = await validate_model_io(simple_train_function, simple_predict_function, laos_monthly_data)
        assert isinstance(result.success, bool)

    async def test_errors_is_list(
        self,
        simple_train_function: TrainFunction,
        simple_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that errors is a list."""
        result = await validate_model_io(simple_train_function, simple_predict_function, laos_monthly_data)
        assert isinstance(result.errors, list)

    async def test_warnings_is_list(
        self,
        simple_train_function: TrainFunction,
        simple_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that warnings is a list."""
        result = await validate_model_io(simple_train_function, simple_predict_function, laos_monthly_data)
        assert isinstance(result.warnings, list)

    async def test_n_predictions_is_int(
        self,
        simple_train_function: TrainFunction,
        simple_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that n_predictions is an int."""
        result = await validate_model_io(simple_train_function, simple_predict_function, laos_monthly_data)
        assert isinstance(result.n_predictions, int)

    async def test_n_samples_is_int(
        self,
        simple_train_function: TrainFunction,
        simple_predict_function: PredictFunction,
        laos_monthly_data: ExampleData,
    ) -> None:
        """Test that n_samples is an int."""
        result = await validate_model_io(simple_train_function, simple_predict_function, laos_monthly_data)
        assert isinstance(result.n_samples, int)
