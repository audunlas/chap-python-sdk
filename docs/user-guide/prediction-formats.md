# Prediction Formats

The SDK supports multiple prediction formats with conversion utilities.

## Format Types

### Nested Format (Internal)

The internal format uses a DataFrame with a `samples` column containing lists:

| time_period | location | samples |
|-------------|----------|---------|
| 2013-04 | Bokeo | [9, 5, 46, ..., 5] |
| 2013-05 | Bokeo | [12, 0, 43, ..., 17] |

This is the format returned by model predict functions.

### Wide Format (CSV)

Wide format has separate columns for each sample:

```csv
time_period,location,sample_0,sample_1,sample_2,...,sample_999
2013-04,Bokeo,9,5,46,...,5
2013-05,Bokeo,12,0,43,...,17
```

This is used for CHAP CSV output.

### Long Format (scoringutils)

Long format has a row for each sample:

| time_period | location | sample_id | prediction |
|-------------|----------|-----------|------------|
| 2013-04 | Bokeo | 0 | 9 |
| 2013-04 | Bokeo | 1 | 5 |
| 2013-04 | Bokeo | 2 | 46 |

This is used for scoringutils integration.

## Format Detection

Automatically detect the format of a DataFrame:

```python
from chap_python_sdk.testing import detect_prediction_format

format_type = detect_prediction_format(dataframe)
# Returns: "nested", "wide", or "long"
```

## Format Conversions

### Nested to Wide

```python
from chap_python_sdk.testing import predictions_to_wide

# Convert from nested to wide (for CHAP CSV output)
wide_predictions = predictions_to_wide(predictions)
```

### Wide to Nested

```python
from chap_python_sdk.testing import predictions_from_wide

# Convert from wide to nested (from CHAP CSV)
nested_predictions = predictions_from_wide(wide_dataframe)
```

### Nested to Long

```python
from chap_python_sdk.testing import predictions_to_long

# Convert from nested to long (for scoringutils)
long_predictions = predictions_to_long(predictions)
```

### Long to Nested

```python
from chap_python_sdk.testing import predictions_from_long

# Convert from long to nested
nested_predictions = predictions_from_long(long_dataframe)
```

## Checking for Samples

Check if a DataFrame has valid prediction samples:

```python
from chap_python_sdk.testing import has_prediction_samples

if has_prediction_samples(dataframe):
    print("DataFrame has valid samples column")
```

## Statistics and Quantiles

### Generate Quantiles

```python
from chap_python_sdk.testing import predictions_to_quantiles

quantiles_df = predictions_to_quantiles(
    predictions,
    quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]
)
```

### Summary Statistics

```python
from chap_python_sdk.testing import predictions_summary

summary = predictions_summary(predictions)
# Returns DataFrame with mean, median, and confidence intervals
```

## Example Workflow

```python
from chap_python_sdk.testing import (
    detect_prediction_format,
    predictions_from_wide,
    predictions_to_long,
)

# Load predictions from CSV (wide format)
import pandas as pd
wide_df = pd.read_csv("predictions.csv")

# Check format
print(detect_prediction_format(wide_df))  # "wide"

# Convert to nested for processing
nested = predictions_from_wide(wide_df)

# Convert to long for scoringutils
long_df = predictions_to_long(nested)
```

## Next Steps

- Learn about [Assertions](assertions.md)
- Check out the [API Reference](../api/predictions.md)
