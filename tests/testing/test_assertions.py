"""Tests for assertion helpers."""

import pytest
from chapkit.data import DataFrame

from chap_python_sdk.testing.assertions import (
    PredictionValidationError,
    assert_consistent_sample_counts,
    assert_numeric_samples,
    assert_prediction_shape,
    assert_samples_column,
    assert_time_location_columns,
    assert_valid_predictions,
)


class TestAssertValidPredictions:
    """Tests for assert_valid_predictions function."""

    def test_passes_for_valid_predictions(self, valid_nested_predictions: DataFrame) -> None:
        """Test that valid predictions pass validation."""
        assert_valid_predictions(valid_nested_predictions)

    def test_passes_with_expected_rows(self, valid_nested_predictions: DataFrame) -> None:
        """Test validation with expected row count."""
        assert_valid_predictions(valid_nested_predictions, expected_rows=4)

    def test_raises_for_wrong_row_count(self, valid_nested_predictions: DataFrame) -> None:
        """Test that wrong row count raises error."""
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_valid_predictions(valid_nested_predictions, expected_rows=10)
        assert "Expected 10 prediction rows" in str(exception_info.value)

    def test_raises_for_empty_dataframe(self) -> None:
        """Test that empty DataFrame raises error."""
        dataframe = DataFrame.from_dict({"time_period": [], "location": [], "samples": []})
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_valid_predictions(dataframe)
        assert "empty" in str(exception_info.value)

    def test_raises_for_non_dataframe(self) -> None:
        """Test that non-DataFrame input raises error."""
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_valid_predictions({"not": "a dataframe"})  # type: ignore[arg-type]
        assert "DataFrame" in str(exception_info.value)


class TestAssertPredictionShape:
    """Tests for assert_prediction_shape function."""

    def test_passes_for_matching_shape(
        self, valid_nested_predictions: DataFrame, sample_future_data: DataFrame
    ) -> None:
        """Test that matching shape passes."""
        assert_prediction_shape(valid_nested_predictions, sample_future_data)

    def test_raises_for_row_mismatch(
        self, valid_nested_predictions: DataFrame, sample_training_data: DataFrame
    ) -> None:
        """Test that row count mismatch raises error."""
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_prediction_shape(valid_nested_predictions, sample_training_data)
        assert "rows" in str(exception_info.value)

    def test_raises_for_missing_time_periods(self, sample_future_data: DataFrame) -> None:
        """Test that missing time periods raise error."""
        predictions = DataFrame.from_dict(
            {
                "time_period": ["2023-04", "2023-04", "2023-04", "2023-04"],
                "location": ["A", "A", "B", "B"],
                "samples": [[1.0], [2.0], [3.0], [4.0]],
            }
        )
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_prediction_shape(predictions, sample_future_data)
        assert "Time periods" in str(exception_info.value)

    def test_raises_for_missing_locations(self, sample_future_data: DataFrame) -> None:
        """Test that missing locations raise error."""
        predictions = DataFrame.from_dict(
            {
                "time_period": ["2023-04", "2023-05", "2023-04", "2023-05"],
                "location": ["A", "A", "A", "A"],
                "samples": [[1.0], [2.0], [3.0], [4.0]],
            }
        )
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_prediction_shape(predictions, sample_future_data)
        assert "Locations" in str(exception_info.value)


class TestAssertSamplesColumn:
    """Tests for assert_samples_column function."""

    def test_passes_for_valid_samples(self, valid_nested_predictions: DataFrame) -> None:
        """Test that valid samples pass."""
        assert_samples_column(valid_nested_predictions)

    def test_raises_without_samples_column(self) -> None:
        """Test that missing samples column raises error."""
        dataframe = DataFrame.from_dict({"time_period": ["2023-01"], "location": ["A"]})
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_samples_column(dataframe)
        assert "samples" in str(exception_info.value)

    def test_raises_for_non_list_samples(self) -> None:
        """Test that non-list samples raise error."""
        dataframe = DataFrame.from_dict(
            {
                "time_period": ["2023-01"],
                "location": ["A"],
                "samples": [42],
            }
        )
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_samples_column(dataframe)
        assert "list" in str(exception_info.value)

    def test_raises_for_too_few_samples(self, valid_nested_predictions: DataFrame) -> None:
        """Test that too few samples raise error."""
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_samples_column(valid_nested_predictions, min_samples=10)
        assert "at least 10 samples" in str(exception_info.value)

    def test_raises_for_too_many_samples(self, valid_nested_predictions: DataFrame) -> None:
        """Test that too many samples raise error."""
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_samples_column(valid_nested_predictions, max_samples=2)
        assert "at most 2 samples" in str(exception_info.value)


class TestAssertConsistentSampleCounts:
    """Tests for assert_consistent_sample_counts function."""

    def test_passes_for_consistent_counts(self, valid_nested_predictions: DataFrame) -> None:
        """Test that consistent sample counts pass."""
        assert_consistent_sample_counts(valid_nested_predictions)

    def test_raises_for_inconsistent_counts(self) -> None:
        """Test that inconsistent sample counts raise error."""
        dataframe = DataFrame.from_dict(
            {
                "time_period": ["2023-01", "2023-02"],
                "location": ["A", "A"],
                "samples": [[1.0, 2.0], [1.0, 2.0, 3.0]],
            }
        )
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_consistent_sample_counts(dataframe)
        assert "Inconsistent sample counts" in str(exception_info.value)

    def test_raises_without_samples_column(self) -> None:
        """Test that missing samples column raises error."""
        dataframe = DataFrame.from_dict({"time_period": ["2023-01"], "location": ["A"]})
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_consistent_sample_counts(dataframe)
        assert "samples" in str(exception_info.value)

    def test_passes_for_empty_dataframe(self) -> None:
        """Test that empty DataFrame passes."""
        dataframe = DataFrame.from_dict({"time_period": [], "location": [], "samples": []})
        assert_consistent_sample_counts(dataframe)


class TestAssertNumericSamples:
    """Tests for assert_numeric_samples function."""

    def test_passes_for_numeric_samples(self, valid_nested_predictions: DataFrame) -> None:
        """Test that numeric samples pass."""
        assert_numeric_samples(valid_nested_predictions)

    def test_raises_for_string_samples(self) -> None:
        """Test that string samples raise error."""
        dataframe = DataFrame.from_dict(
            {
                "time_period": ["2023-01"],
                "location": ["A"],
                "samples": [["not", "numeric"]],
            }
        )
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_numeric_samples(dataframe)
        assert "numeric value" in str(exception_info.value)

    def test_raises_for_nan_samples(self) -> None:
        """Test that NaN samples raise error."""
        dataframe = DataFrame.from_dict(
            {
                "time_period": ["2023-01"],
                "location": ["A"],
                "samples": [[1.0, float("nan")]],
            }
        )
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_numeric_samples(dataframe)
        assert "nan" in str(exception_info.value).lower()

    def test_raises_for_inf_samples(self) -> None:
        """Test that infinity samples raise error."""
        dataframe = DataFrame.from_dict(
            {
                "time_period": ["2023-01"],
                "location": ["A"],
                "samples": [[1.0, float("inf")]],
            }
        )
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_numeric_samples(dataframe)
        assert "inf" in str(exception_info.value).lower()

    def test_raises_without_samples_column(self) -> None:
        """Test that missing samples column raises error."""
        dataframe = DataFrame.from_dict({"time_period": ["2023-01"], "location": ["A"]})
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_numeric_samples(dataframe)
        assert "samples" in str(exception_info.value)


class TestAssertTimeLocationColumns:
    """Tests for assert_time_location_columns function."""

    def test_passes_for_valid_columns(self, valid_nested_predictions: DataFrame) -> None:
        """Test that valid columns pass."""
        assert_time_location_columns(valid_nested_predictions)

    def test_raises_without_time_period(self) -> None:
        """Test that missing time_period raises error."""
        dataframe = DataFrame.from_dict({"location": ["A"], "samples": [[1.0]]})
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_time_location_columns(dataframe)
        assert "time_period" in str(exception_info.value)

    def test_raises_without_location(self) -> None:
        """Test that missing location raises error."""
        dataframe = DataFrame.from_dict({"time_period": ["2023-01"], "samples": [[1.0]]})
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_time_location_columns(dataframe)
        assert "location" in str(exception_info.value)

    def test_raises_for_both_missing(self) -> None:
        """Test that both missing columns are reported."""
        dataframe = DataFrame.from_dict({"samples": [[1.0]]})
        with pytest.raises(PredictionValidationError) as exception_info:
            assert_time_location_columns(dataframe)
        error_message = str(exception_info.value)
        assert "time_period" in error_message
        assert "location" in error_message
