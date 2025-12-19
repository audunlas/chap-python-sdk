"""Tests for example data loading functionality."""

import pytest

from chap_python_sdk.testing import ExampleData, get_example_data, list_available_datasets


class TestListAvailableDatasets:
    """Tests for list_available_datasets function."""

    def test_returns_list(self) -> None:
        """Test that list_available_datasets returns a list."""
        datasets = list_available_datasets()
        assert isinstance(datasets, list)

    def test_returns_tuples(self) -> None:
        """Test that each dataset is a (country, frequency) tuple."""
        datasets = list_available_datasets()
        for dataset in datasets:
            assert isinstance(dataset, tuple)
            assert len(dataset) == 2

    def test_contains_laos_monthly(self) -> None:
        """Test that Laos monthly dataset is available."""
        datasets = list_available_datasets()
        assert ("laos", "monthly") in datasets


class TestGetExampleData:
    """Tests for get_example_data function."""

    def test_loads_laos_monthly(self) -> None:
        """Test loading Laos monthly dataset."""
        data = get_example_data(country="laos", frequency="monthly")
        assert isinstance(data, ExampleData)

    def test_returns_example_data_instance(self, laos_monthly_data: ExampleData) -> None:
        """Test that get_example_data returns an ExampleData instance."""
        assert isinstance(laos_monthly_data, ExampleData)

    def test_training_data_not_empty(self, laos_monthly_data: ExampleData) -> None:
        """Test that training data is not empty."""
        assert len(laos_monthly_data.training_data) > 0

    def test_historic_data_not_empty(self, laos_monthly_data: ExampleData) -> None:
        """Test that historic data is not empty."""
        assert len(laos_monthly_data.historic_data) > 0

    def test_future_data_not_empty(self, laos_monthly_data: ExampleData) -> None:
        """Test that future data is not empty."""
        assert len(laos_monthly_data.future_data) > 0

    def test_training_data_has_time_period(self, laos_monthly_data: ExampleData) -> None:
        """Test that training data has time_period column."""
        assert "time_period" in laos_monthly_data.training_data.columns

    def test_training_data_has_location(self, laos_monthly_data: ExampleData) -> None:
        """Test that training data has location column."""
        assert "location" in laos_monthly_data.training_data.columns

    def test_historic_data_has_time_period(self, laos_monthly_data: ExampleData) -> None:
        """Test that historic data has time_period column."""
        assert "time_period" in laos_monthly_data.historic_data.columns

    def test_historic_data_has_location(self, laos_monthly_data: ExampleData) -> None:
        """Test that historic data has location column."""
        assert "location" in laos_monthly_data.historic_data.columns

    def test_future_data_has_time_period(self, laos_monthly_data: ExampleData) -> None:
        """Test that future data has time_period column."""
        assert "time_period" in laos_monthly_data.future_data.columns

    def test_future_data_has_location(self, laos_monthly_data: ExampleData) -> None:
        """Test that future data has location column."""
        assert "location" in laos_monthly_data.future_data.columns

    def test_invalid_country_raises_error(self) -> None:
        """Test that invalid country raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exception_info:
            get_example_data(country="invalid_country", frequency="monthly")
        assert "Dataset not found" in str(exception_info.value)

    def test_invalid_frequency_raises_error(self) -> None:
        """Test that invalid frequency raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exception_info:
            get_example_data(country="laos", frequency="invalid_frequency")
        assert "Dataset not found" in str(exception_info.value)

    def test_error_message_shows_available_datasets(self) -> None:
        """Test that error message includes available datasets."""
        with pytest.raises(FileNotFoundError) as exception_info:
            get_example_data(country="invalid", frequency="invalid")
        assert "Available datasets:" in str(exception_info.value)

    def test_case_insensitive_country(self) -> None:
        """Test that country lookup is case insensitive."""
        data_lower = get_example_data(country="laos", frequency="monthly")
        data_upper = get_example_data(country="LAOS", frequency="monthly")
        assert len(data_lower.training_data) == len(data_upper.training_data)

    def test_case_insensitive_frequency(self) -> None:
        """Test that frequency lookup is case insensitive."""
        data_lower = get_example_data(country="laos", frequency="monthly")
        data_upper = get_example_data(country="laos", frequency="MONTHLY")
        assert len(data_lower.training_data) == len(data_upper.training_data)

    def test_predictions_loaded_if_exists(self, laos_monthly_data: ExampleData) -> None:
        """Test that predictions are loaded if the file exists."""
        assert laos_monthly_data.predictions is not None

    def test_configuration_is_none_by_default(self, laos_monthly_data: ExampleData) -> None:
        """Test that configuration is None by default."""
        assert laos_monthly_data.configuration is None

    def test_configuration_passed_through(self) -> None:
        """Test that configuration parameter is passed through."""
        config = {"learning_rate": 0.01}
        data = get_example_data(country="laos", frequency="monthly", configuration=config)
        assert data.configuration == config

    def test_geo_is_none(self, laos_monthly_data: ExampleData) -> None:
        """Test that geo is None when no geo file exists."""
        assert laos_monthly_data.geo is None
