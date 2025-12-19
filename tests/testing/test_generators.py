"""Tests for dataset generators."""

import pytest

from chap_python_sdk.testing import (
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
        run_info = generate_run_info(service_info, prediction_length=6)

        assert run_info.prediction_length == 6

    def test_additional_covariates_allowed(self) -> None:
        """Test additional covariates when allowed."""
        service_info = MLServiceInfo(allow_free_additional_continuous_covariates=True)
        run_info = generate_run_info(
            service_info, additional_covariates=["humidity", "wind_speed"]
        )

        assert run_info.additional_continuous_covariates == ["humidity", "wind_speed"]

    def test_additional_covariates_not_allowed(self) -> None:
        """Test error when additional covariates not allowed."""
        service_info = MLServiceInfo(allow_free_additional_continuous_covariates=False)

        with pytest.raises(ValueError, match="does not allow additional covariates"):
            generate_run_info(service_info, additional_covariates=["humidity"])

    def test_empty_covariates_always_allowed(self) -> None:
        """Test that empty covariates list is always allowed."""
        service_info = MLServiceInfo(allow_free_additional_continuous_covariates=False)
        run_info = generate_run_info(service_info, additional_covariates=[])

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

        example_data = generate_example_data(service_info, run_info, seed=42)

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
        service_info = MLServiceInfo(
            required_covariates=["rainfall", "mean_temperature"]
        )
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

        example_data = generate_example_data(service_info, run_info, n_locations=5)
        training_df = example_data.training_data.to_pandas()

        assert training_df["location"].nunique() == 5

    def test_custom_location_names(self) -> None:
        """Test custom location names."""
        service_info = MLServiceInfo()
        run_info = RunInfo(prediction_length=3)
        locations = ["Oslo", "Bergen", "Trondheim"]

        example_data = generate_example_data(
            service_info, run_info, location_names=locations
        )
        training_df = example_data.training_data.to_pandas()

        assert set(training_df["location"].unique()) == set(locations)

    def test_future_data_row_count(self) -> None:
        """Test future data has correct number of rows."""
        service_info = MLServiceInfo()
        run_info = RunInfo(prediction_length=4)

        example_data = generate_example_data(service_info, run_info, n_locations=3)
        future_df = example_data.future_data.to_pandas()

        expected_rows = 4 * 3
        assert len(future_df) == expected_rows

    def test_reproducibility_with_seed(self) -> None:
        """Test that seed produces reproducible results."""
        service_info = MLServiceInfo(required_covariates=["rainfall"])
        run_info = RunInfo(prediction_length=3)

        data1 = generate_example_data(service_info, run_info, seed=42)
        data2 = generate_example_data(service_info, run_info, seed=42)

        df1 = data1.training_data.to_pandas()
        df2 = data2.training_data.to_pandas()

        assert df1.equals(df2)

    def test_monthly_period_format(self) -> None:
        """Test monthly period format."""
        service_info = MLServiceInfo(supported_period_type=PeriodType.month)
        run_info = RunInfo(prediction_length=3)

        example_data = generate_example_data(service_info, run_info, n_locations=1)
        training_df = example_data.training_data.to_pandas()

        first_period = training_df["time_period"].iloc[0]
        assert "-" in first_period
        assert len(first_period.split("-")) == 2

    def test_weekly_period_format(self) -> None:
        """Test weekly period format."""
        service_info = MLServiceInfo(supported_period_type=PeriodType.week)
        run_info = RunInfo(prediction_length=3)

        example_data = generate_example_data(service_info, run_info, n_locations=1)
        training_df = example_data.training_data.to_pandas()

        first_period = training_df["time_period"].iloc[0]
        assert "/" in first_period

    def test_yearly_period_format(self) -> None:
        """Test yearly period format."""
        service_info = MLServiceInfo(supported_period_type=PeriodType.year)
        run_info = RunInfo(prediction_length=2)

        example_data = generate_example_data(service_info, run_info, n_locations=1)
        training_df = example_data.training_data.to_pandas()

        first_period = training_df["time_period"].iloc[0]
        assert first_period.isdigit()
        assert len(first_period) == 4


class TestGenerateTestData:
    """Tests for generate_test_data convenience function."""

    def test_combines_run_info_and_example_data(self) -> None:
        """Test that generate_test_data combines both steps."""
        service_info = MLServiceInfo(
            required_covariates=["rainfall"],
            supported_period_type=PeriodType.month,
        )

        example_data = generate_test_data(service_info, prediction_length=5, seed=42)

        assert example_data.run_info is not None
        assert example_data.run_info.prediction_length == 5
        assert example_data.training_data is not None

    def test_passes_all_parameters(self) -> None:
        """Test that all parameters are passed through."""
        service_info = MLServiceInfo(allow_free_additional_continuous_covariates=True)
        locations = ["A", "B"]

        example_data = generate_test_data(
            service_info,
            prediction_length=2,
            additional_covariates=["humidity"],
            n_locations=2,
            location_names=locations,
            seed=123,
        )

        assert example_data.run_info is not None
        assert example_data.run_info.prediction_length == 2
        assert "humidity" in example_data.run_info.additional_continuous_covariates

        training_df = example_data.training_data.to_pandas()
        assert set(training_df["location"].unique()) == set(locations)
        assert "humidity" in training_df.columns
