"""Tests for dataset generators."""

import pytest

from chap_python_sdk.testing import (
    DataGenerationConfig,
    ExampleData,
    MLServiceInfo,
    PeriodType,
    RunInfo,
    generate_example_data,
    generate_run_info,
    generate_test_data,
)


class TestGenerateRunInfo:
    """Tests for generate_run_info function."""

    def test_default_prediction_length_monthly(self) -> None:
        """Test default prediction length for monthly data."""
        service_info = MLServiceInfo(supported_period_type=PeriodType.month)
        run_info = generate_run_info(service_info)

        assert run_info.prediction_length == 3

    def test_default_prediction_length_weekly(self) -> None:
        """Test default prediction length for weekly data."""
        service_info = MLServiceInfo(supported_period_type=PeriodType.week)
        run_info = generate_run_info(service_info)

        assert run_info.prediction_length == 4

    def test_default_prediction_length_yearly(self) -> None:
        """Test default prediction length for yearly data."""
        service_info = MLServiceInfo(supported_period_type=PeriodType.year)
        run_info = generate_run_info(service_info)

        assert run_info.prediction_length == 1

    def test_custom_prediction_length(self) -> None:
        """Test custom prediction length override."""
        service_info = MLServiceInfo(supported_period_type=PeriodType.month)
        config = DataGenerationConfig(prediction_length=6)
        run_info = generate_run_info(service_info, config)

        assert run_info.prediction_length == 6

    def test_additional_covariates_allowed(self) -> None:
        """Test additional covariates when allowed."""
        service_info = MLServiceInfo(allow_free_additional_continuous_covariates=True)
        config = DataGenerationConfig(additional_covariates=["humidity", "wind_speed"])
        run_info = generate_run_info(service_info, config)

        assert run_info.additional_continuous_covariates == ["humidity", "wind_speed"]

    def test_additional_covariates_not_allowed(self) -> None:
        """Test error when additional covariates not allowed."""
        service_info = MLServiceInfo(allow_free_additional_continuous_covariates=False)
        config = DataGenerationConfig(additional_covariates=["humidity"])

        with pytest.raises(ValueError, match="does not allow additional covariates"):
            generate_run_info(service_info, config)

    def test_empty_covariates_always_allowed(self) -> None:
        """Test that empty covariates list is always allowed."""
        service_info = MLServiceInfo(allow_free_additional_continuous_covariates=False)
        config = DataGenerationConfig(additional_covariates=[])
        run_info = generate_run_info(service_info, config)

        assert run_info.additional_continuous_covariates == []


class TestGenerateExampleData:
    """Tests for generate_example_data function."""

    def test_basic_generation(self) -> None:
        """Test basic example data generation."""
        service_info = MLServiceInfo(
            required_covariates=["rainfall", "mean_temperature"],
            supported_period_type=PeriodType.month,
        )
        run_info = RunInfo(prediction_length=3)
        config = DataGenerationConfig(seed=42)

        example_data = generate_example_data(service_info, run_info, config)

        assert isinstance(example_data, ExampleData)
        assert example_data.training_data is not None
        assert example_data.historic_data is not None
        assert example_data.future_data is not None
        assert example_data.run_info == run_info

    def test_training_data_has_target(self) -> None:
        """Test that training data includes disease_cases."""
        service_info = MLServiceInfo()
        run_info = RunInfo(prediction_length=3)

        example_data = generate_example_data(service_info, run_info)
        training_df = example_data.training_data.to_pandas()

        assert "disease_cases" in training_df.columns
        assert "time_period" in training_df.columns
        assert "location" in training_df.columns
        assert "population" in training_df.columns

    def test_future_data_no_target(self) -> None:
        """Test that future data excludes disease_cases."""
        service_info = MLServiceInfo()
        run_info = RunInfo(prediction_length=3)

        example_data = generate_example_data(service_info, run_info)
        future_df = example_data.future_data.to_pandas()

        assert "disease_cases" not in future_df.columns
        assert "time_period" in future_df.columns
        assert "location" in future_df.columns

    def test_required_covariates_present(self) -> None:
        """Test that required covariates are in all dataframes."""
        service_info = MLServiceInfo(required_covariates=["rainfall", "mean_temperature"])
        run_info = RunInfo(prediction_length=3)

        example_data = generate_example_data(service_info, run_info)

        for df in [
            example_data.training_data.to_pandas(),
            example_data.historic_data.to_pandas(),
            example_data.future_data.to_pandas(),
        ]:
            assert "rainfall" in df.columns
            assert "mean_temperature" in df.columns

    def test_location_count(self) -> None:
        """Test correct number of locations."""
        service_info = MLServiceInfo()
        run_info = RunInfo(prediction_length=3)
        config = DataGenerationConfig(n_locations=5)

        example_data = generate_example_data(service_info, run_info, config)
        training_df = example_data.training_data.to_pandas()

        assert training_df["location"].nunique() == 5

    def test_custom_location_names(self) -> None:
        """Test custom location names."""
        service_info = MLServiceInfo()
        run_info = RunInfo(prediction_length=3)
        locations = ["Oslo", "Bergen", "Trondheim"]
        config = DataGenerationConfig(location_names=locations)

        example_data = generate_example_data(service_info, run_info, config)
        training_df = example_data.training_data.to_pandas()

        assert set(training_df["location"].unique()) == set(locations)

    def test_future_data_row_count(self) -> None:
        """Test future data has correct number of rows."""
        service_info = MLServiceInfo()
        run_info = RunInfo(prediction_length=4)
        config = DataGenerationConfig(n_locations=3)

        example_data = generate_example_data(service_info, run_info, config)
        future_df = example_data.future_data.to_pandas()

        expected_rows = 4 * 3
        assert len(future_df) == expected_rows

    def test_reproducibility_with_seed(self) -> None:
        """Test that seed produces reproducible results."""
        service_info = MLServiceInfo(required_covariates=["rainfall"])
        run_info = RunInfo(prediction_length=3)

        data1 = generate_example_data(service_info, run_info, DataGenerationConfig(seed=42))
        data2 = generate_example_data(service_info, run_info, DataGenerationConfig(seed=42))

        df1 = data1.training_data.to_pandas()
        df2 = data2.training_data.to_pandas()

        assert df1.equals(df2)

    def test_monthly_period_format(self) -> None:
        """Test monthly period format."""
        service_info = MLServiceInfo(supported_period_type=PeriodType.month)
        run_info = RunInfo(prediction_length=3)
        config = DataGenerationConfig(n_locations=1)

        example_data = generate_example_data(service_info, run_info, config)
        training_df = example_data.training_data.to_pandas()

        first_period = training_df["time_period"].iloc[0]
        assert "-" in first_period
        assert len(first_period.split("-")) == 2

    def test_weekly_period_format(self) -> None:
        """Test weekly period format."""
        service_info = MLServiceInfo(supported_period_type=PeriodType.week)
        run_info = RunInfo(prediction_length=3)
        config = DataGenerationConfig(n_locations=1)

        example_data = generate_example_data(service_info, run_info, config)
        training_df = example_data.training_data.to_pandas()

        first_period = training_df["time_period"].iloc[0]
        assert "/" in first_period

    def test_yearly_period_format(self) -> None:
        """Test yearly period format."""
        service_info = MLServiceInfo(supported_period_type=PeriodType.year)
        run_info = RunInfo(prediction_length=2)
        config = DataGenerationConfig(n_locations=1)

        example_data = generate_example_data(service_info, run_info, config)
        training_df = example_data.training_data.to_pandas()

        first_period = training_df["time_period"].iloc[0]
        assert first_period.isdigit()
        assert len(first_period) == 4


class TestIncludeNans:
    """Tests for NaN injection feature."""

    def test_include_nans_adds_missing_values(self) -> None:
        """Test that include_nans adds NaN values to data."""
        service_info = MLServiceInfo(required_covariates=["rainfall"])
        config = DataGenerationConfig(
            include_nans=True,
            nan_fraction=0.2,
            seed=42,
            n_locations=3,
            n_training_periods=10,
        )

        example_data = generate_test_data(service_info, config)
        training_df = example_data.training_data.to_pandas()

        assert training_df["rainfall"].isna().any(), "Expected NaN values in rainfall column"

    def test_no_nans_by_default(self) -> None:
        """Test that NaN values are not added by default."""
        service_info = MLServiceInfo(required_covariates=["rainfall"])
        config = DataGenerationConfig(seed=42)

        example_data = generate_test_data(service_info, config)
        training_df = example_data.training_data.to_pandas()

        assert not training_df["rainfall"].isna().any(), "Unexpected NaN values found"

    def test_nans_in_all_dataframes(self) -> None:
        """Test that NaN values are added to all dataframes."""
        service_info = MLServiceInfo(required_covariates=["rainfall"])
        config = DataGenerationConfig(
            include_nans=True,
            nan_fraction=0.3,
            seed=42,
            n_locations=5,
            n_training_periods=20,
        )

        example_data = generate_test_data(service_info, config)

        for name, df in [
            ("training", example_data.training_data.to_pandas()),
            ("historic", example_data.historic_data.to_pandas()),
            ("future", example_data.future_data.to_pandas()),
        ]:
            assert df["rainfall"].isna().any(), f"Expected NaN values in {name} data"

    def test_nan_fraction_zero_no_nans(self) -> None:
        """Test that nan_fraction=0 adds no NaN values."""
        service_info = MLServiceInfo(required_covariates=["rainfall"])
        config = DataGenerationConfig(include_nans=True, nan_fraction=0.0, seed=42)

        example_data = generate_test_data(service_info, config)
        training_df = example_data.training_data.to_pandas()

        assert not training_df["rainfall"].isna().any()

    def test_time_period_and_location_never_nan(self) -> None:
        """Test that time_period and location columns never have NaN values."""
        service_info = MLServiceInfo(required_covariates=["rainfall"])
        config = DataGenerationConfig(include_nans=True, nan_fraction=0.5, seed=42)

        example_data = generate_test_data(service_info, config)

        for df in [
            example_data.training_data.to_pandas(),
            example_data.historic_data.to_pandas(),
            example_data.future_data.to_pandas(),
        ]:
            assert not df["time_period"].isna().any()
            assert not df["location"].isna().any()

    def test_nan_fraction_approximate(self) -> None:
        """Test that nan_fraction approximately controls NaN percentage."""
        service_info = MLServiceInfo(required_covariates=["rainfall"])
        config = DataGenerationConfig(
            include_nans=True,
            nan_fraction=0.2,
            seed=42,
            n_locations=10,
            n_training_periods=50,
        )

        example_data = generate_test_data(service_info, config)
        training_df = example_data.training_data.to_pandas()

        nan_fraction_actual = training_df["rainfall"].isna().mean()
        assert 0.1 < nan_fraction_actual < 0.3, f"Expected ~20% NaN, got {nan_fraction_actual:.1%}"


class TestGenerateTestData:
    """Tests for generate_test_data convenience function."""

    def test_combines_run_info_and_example_data(self) -> None:
        """Test that generate_test_data combines both steps."""
        service_info = MLServiceInfo(
            required_covariates=["rainfall"],
            supported_period_type=PeriodType.month,
        )
        config = DataGenerationConfig(prediction_length=5, seed=42)

        example_data = generate_test_data(service_info, config)

        assert example_data.run_info is not None
        assert example_data.run_info.prediction_length == 5
        assert example_data.training_data is not None

    def test_passes_all_parameters(self) -> None:
        """Test that all parameters are passed through."""
        service_info = MLServiceInfo(allow_free_additional_continuous_covariates=True)
        locations = ["A", "B"]
        config = DataGenerationConfig(
            prediction_length=2,
            additional_covariates=["humidity"],
            n_locations=2,
            location_names=locations,
            seed=123,
        )

        example_data = generate_test_data(service_info, config)

        assert example_data.run_info is not None
        assert example_data.run_info.prediction_length == 2
        assert "humidity" in example_data.run_info.additional_continuous_covariates

        training_df = example_data.training_data.to_pandas()
        assert set(training_df["location"].unique()) == set(locations)
        assert "humidity" in training_df.columns


class TestDataGenerationConfig:
    """Tests for DataGenerationConfig Pydantic model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DataGenerationConfig()

        assert config.prediction_length is None
        assert config.additional_covariates == []
        assert config.n_locations == 3
        assert config.n_training_periods == 24
        assert config.n_historic_periods == 12
        assert config.seed is None
        assert config.location_names is None
        assert config.include_nans is False
        assert config.nan_fraction == 0.1

    def test_validation_n_locations_positive(self) -> None:
        """Test that n_locations must be positive."""
        with pytest.raises(ValueError):
            DataGenerationConfig(n_locations=0)

    def test_validation_nan_fraction_range(self) -> None:
        """Test that nan_fraction must be between 0 and 1."""
        with pytest.raises(ValueError):
            DataGenerationConfig(nan_fraction=-0.1)

        with pytest.raises(ValueError):
            DataGenerationConfig(nan_fraction=1.5)

    def test_model_dump(self) -> None:
        """Test that config can be serialized."""
        config = DataGenerationConfig(
            prediction_length=5,
            include_nans=True,
            nan_fraction=0.2,
        )

        data = config.model_dump()

        assert data["prediction_length"] == 5
        assert data["include_nans"] is True
        assert data["nan_fraction"] == 0.2


class TestMLServiceInfo:
    """Tests for MLServiceInfo Pydantic model."""

    def test_default_values(self) -> None:
        """Test default MLServiceInfo values."""
        info = MLServiceInfo()

        assert info.required_covariates == []
        assert info.allow_free_additional_continuous_covariates is False
        assert info.supported_period_type == PeriodType.any

    def test_with_values(self) -> None:
        """Test MLServiceInfo with custom values."""
        info = MLServiceInfo(
            required_covariates=["rainfall", "temperature"],
            allow_free_additional_continuous_covariates=True,
            supported_period_type=PeriodType.week,
        )

        assert info.required_covariates == ["rainfall", "temperature"]
        assert info.allow_free_additional_continuous_covariates is True
        assert info.supported_period_type == PeriodType.week

    def test_period_type_from_string(self) -> None:
        """Test that period type can be created from string."""
        info = MLServiceInfo(supported_period_type="month")  # type: ignore[arg-type]

        assert info.supported_period_type == PeriodType.month
