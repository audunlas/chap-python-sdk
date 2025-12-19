"""Tests for prediction format utilities."""

from typing import Any, cast

import pytest
from chapkit.data import DataFrame

from chap_python_sdk.testing import (
    detect_prediction_format,
    has_prediction_samples,
    predictions_from_long,
    predictions_from_wide,
    predictions_summary,
    predictions_to_long,
    predictions_to_quantiles,
    predictions_to_wide,
)


class TestDetectPredictionFormat:
    """Tests for detect_prediction_format function."""

    def test_detects_nested_format(self, valid_nested_predictions: DataFrame) -> None:
        """Test detection of nested format."""
        assert detect_prediction_format(valid_nested_predictions) == "nested"

    def test_detects_wide_format(self, valid_wide_predictions: DataFrame) -> None:
        """Test detection of wide format."""
        assert detect_prediction_format(valid_wide_predictions) == "wide"

    def test_detects_long_format(self, valid_long_predictions: DataFrame) -> None:
        """Test detection of long format."""
        assert detect_prediction_format(valid_long_predictions) == "long"

    def test_raises_for_unknown_format(self) -> None:
        """Test that unknown format raises ValueError."""
        dataframe = DataFrame.from_dict(
            {
                "time_period": ["2023-01"],
                "location": ["A"],
                "unknown_column": [42],
            }
        )
        with pytest.raises(ValueError) as exception_info:
            detect_prediction_format(dataframe)
        assert "Cannot determine prediction format" in str(exception_info.value)


class TestHasPredictionSamples:
    """Tests for has_prediction_samples function."""

    def test_returns_true_for_valid_samples(self, valid_nested_predictions: DataFrame) -> None:
        """Test that valid samples column returns True."""
        assert has_prediction_samples(valid_nested_predictions) is True

    def test_returns_false_without_samples_column(self, valid_wide_predictions: DataFrame) -> None:
        """Test that missing samples column returns False."""
        assert has_prediction_samples(valid_wide_predictions) is False

    def test_returns_true_for_empty_dataframe_with_samples(self) -> None:
        """Test empty dataframe with samples column returns True."""
        dataframe = DataFrame.from_dict({"samples": []})
        assert has_prediction_samples(dataframe) is True


class TestPredictionsToWide:
    """Tests for predictions_to_wide function."""

    def test_converts_nested_to_wide(self, valid_nested_predictions: DataFrame) -> None:
        """Test conversion from nested to wide format."""
        wide = predictions_to_wide(valid_nested_predictions)
        assert "sample_0" in wide.columns
        assert "sample_1" in wide.columns
        assert "sample_2" in wide.columns
        assert "samples" not in wide.columns

    def test_preserves_non_sample_columns(self, valid_nested_predictions: DataFrame) -> None:
        """Test that non-sample columns are preserved."""
        wide = predictions_to_wide(valid_nested_predictions)
        assert "time_period" in wide.columns
        assert "location" in wide.columns

    def test_preserves_row_count(self, valid_nested_predictions: DataFrame) -> None:
        """Test that row count is preserved."""
        wide = predictions_to_wide(valid_nested_predictions)
        assert len(wide) == len(valid_nested_predictions)

    def test_raises_without_samples_column(self, valid_wide_predictions: DataFrame) -> None:
        """Test that missing samples column raises ValueError."""
        with pytest.raises(ValueError) as exception_info:
            predictions_to_wide(valid_wide_predictions)
        assert "samples" in str(exception_info.value)


class TestPredictionsFromWide:
    """Tests for predictions_from_wide function."""

    def test_converts_wide_to_nested(self, valid_wide_predictions: DataFrame) -> None:
        """Test conversion from wide to nested format."""
        nested = predictions_from_wide(valid_wide_predictions)
        assert "samples" in nested.columns
        assert "sample_0" not in nested.columns

    def test_preserves_non_sample_columns(self, valid_wide_predictions: DataFrame) -> None:
        """Test that non-sample columns are preserved."""
        nested = predictions_from_wide(valid_wide_predictions)
        assert "time_period" in nested.columns
        assert "location" in nested.columns

    def test_preserves_row_count(self, valid_wide_predictions: DataFrame) -> None:
        """Test that row count is preserved."""
        nested = predictions_from_wide(valid_wide_predictions)
        assert len(nested) == len(valid_wide_predictions)

    def test_samples_are_lists(self, valid_wide_predictions: DataFrame) -> None:
        """Test that samples column contains lists."""
        nested = predictions_from_wide(valid_wide_predictions)
        assert isinstance(cast(list[Any], nested["samples"])[0], list)

    def test_raises_without_sample_columns(self, valid_nested_predictions: DataFrame) -> None:
        """Test that missing sample_N columns raises ValueError."""
        dataframe = DataFrame.from_dict(
            {
                "time_period": ["2023-01"],
                "location": ["A"],
            }
        )
        with pytest.raises(ValueError) as exception_info:
            predictions_from_wide(dataframe)
        assert "sample_N" in str(exception_info.value)


class TestNestedWideRoundTrip:
    """Tests for round-trip conversion between nested and wide formats."""

    def test_nested_to_wide_to_nested(self, valid_nested_predictions: DataFrame) -> None:
        """Test round-trip conversion preserves data."""
        wide = predictions_to_wide(valid_nested_predictions)
        nested = predictions_from_wide(wide)

        assert len(nested) == len(valid_nested_predictions)
        assert list(nested["time_period"]) == list(valid_nested_predictions["time_period"])
        assert list(nested["location"]) == list(valid_nested_predictions["location"])
        assert list(nested["samples"]) == list(valid_nested_predictions["samples"])


class TestPredictionsToLong:
    """Tests for predictions_to_long function."""

    def test_converts_nested_to_long(self, valid_nested_predictions: DataFrame) -> None:
        """Test conversion from nested to long format."""
        long_format = predictions_to_long(valid_nested_predictions)
        assert "sample_id" in long_format.columns
        assert "prediction" in long_format.columns
        assert "samples" not in long_format.columns

    def test_expands_rows_correctly(self, valid_nested_predictions: DataFrame) -> None:
        """Test that rows are expanded correctly."""
        long_format = predictions_to_long(valid_nested_predictions)
        n_original_rows = len(valid_nested_predictions)
        n_samples = len(cast(list[Any], valid_nested_predictions["samples"])[0])
        expected_rows = n_original_rows * n_samples
        assert len(long_format) == expected_rows

    def test_raises_without_samples_column(self, valid_wide_predictions: DataFrame) -> None:
        """Test that missing samples column raises ValueError."""
        with pytest.raises(ValueError) as exception_info:
            predictions_to_long(valid_wide_predictions)
        assert "samples" in str(exception_info.value)


class TestPredictionsFromLong:
    """Tests for predictions_from_long function."""

    def test_converts_long_to_nested(self, valid_long_predictions: DataFrame) -> None:
        """Test conversion from long to nested format."""
        nested = predictions_from_long(valid_long_predictions)
        assert "samples" in nested.columns
        assert "sample_id" not in nested.columns
        assert "prediction" not in nested.columns

    def test_raises_without_required_columns(self) -> None:
        """Test that missing required columns raises ValueError."""
        dataframe = DataFrame.from_dict(
            {
                "time_period": ["2023-01"],
                "location": ["A"],
            }
        )
        with pytest.raises(ValueError) as exception_info:
            predictions_from_long(dataframe)
        assert "sample_id" in str(exception_info.value) or "prediction" in str(exception_info.value)


class TestNestedLongRoundTrip:
    """Tests for round-trip conversion between nested and long formats."""

    def test_nested_to_long_to_nested(self, valid_nested_predictions: DataFrame) -> None:
        """Test round-trip conversion preserves data."""
        long_format = predictions_to_long(valid_nested_predictions)
        nested = predictions_from_long(long_format)

        assert len(nested) == len(valid_nested_predictions)


class TestPredictionsToQuantiles:
    """Tests for predictions_to_quantiles function."""

    def test_adds_quantile_columns(self, valid_nested_predictions: DataFrame) -> None:
        """Test that quantile columns are added."""
        quantiles = predictions_to_quantiles(valid_nested_predictions)
        assert "q_0.025" in quantiles.columns
        assert "q_0.5" in quantiles.columns
        assert "q_0.975" in quantiles.columns

    def test_removes_samples_column(self, valid_nested_predictions: DataFrame) -> None:
        """Test that samples column is removed."""
        quantiles = predictions_to_quantiles(valid_nested_predictions)
        assert "samples" not in quantiles.columns

    def test_preserves_row_count(self, valid_nested_predictions: DataFrame) -> None:
        """Test that row count is preserved."""
        quantiles = predictions_to_quantiles(valid_nested_predictions)
        assert len(quantiles) == len(valid_nested_predictions)

    def test_custom_probabilities(self, valid_nested_predictions: DataFrame) -> None:
        """Test with custom probabilities."""
        probabilities = [0.1, 0.5, 0.9]
        quantiles = predictions_to_quantiles(valid_nested_predictions, probabilities=probabilities)
        assert "q_0.1" in quantiles.columns
        assert "q_0.5" in quantiles.columns
        assert "q_0.9" in quantiles.columns

    def test_raises_without_samples_column(self, valid_wide_predictions: DataFrame) -> None:
        """Test that missing samples column raises ValueError."""
        with pytest.raises(ValueError) as exception_info:
            predictions_to_quantiles(valid_wide_predictions)
        assert "samples" in str(exception_info.value)


class TestPredictionsSummary:
    """Tests for predictions_summary function."""

    def test_adds_mean_column(self, valid_nested_predictions: DataFrame) -> None:
        """Test that mean column is added."""
        summary = predictions_summary(valid_nested_predictions)
        assert "mean" in summary.columns

    def test_adds_median_column(self, valid_nested_predictions: DataFrame) -> None:
        """Test that median column is added."""
        summary = predictions_summary(valid_nested_predictions)
        assert "median" in summary.columns

    def test_adds_confidence_intervals(self, valid_nested_predictions: DataFrame) -> None:
        """Test that confidence interval columns are added."""
        summary = predictions_summary(valid_nested_predictions)
        assert "ci_0.5_lower" in summary.columns
        assert "ci_0.5_upper" in summary.columns
        assert "ci_0.9_lower" in summary.columns
        assert "ci_0.9_upper" in summary.columns

    def test_preserves_samples_column(self, valid_nested_predictions: DataFrame) -> None:
        """Test that samples column is preserved."""
        summary = predictions_summary(valid_nested_predictions)
        assert "samples" in summary.columns

    def test_preserves_row_count(self, valid_nested_predictions: DataFrame) -> None:
        """Test that row count is preserved."""
        summary = predictions_summary(valid_nested_predictions)
        assert len(summary) == len(valid_nested_predictions)

    def test_custom_confidence_intervals(self, valid_nested_predictions: DataFrame) -> None:
        """Test with custom confidence intervals."""
        confidence_intervals = [0.8, 0.99]
        summary = predictions_summary(valid_nested_predictions, confidence_intervals=confidence_intervals)
        assert "ci_0.8_lower" in summary.columns
        assert "ci_0.8_upper" in summary.columns
        assert "ci_0.99_lower" in summary.columns
        assert "ci_0.99_upper" in summary.columns

    def test_raises_without_samples_column(self, valid_wide_predictions: DataFrame) -> None:
        """Test that missing samples column raises ValueError."""
        with pytest.raises(ValueError) as exception_info:
            predictions_summary(valid_wide_predictions)
        assert "samples" in str(exception_info.value)
