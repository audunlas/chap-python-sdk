"""Example data loading for model testing."""

from importlib import resources
from pathlib import Path
from typing import Any

from chapkit.data import DataFrame

from chap_python_sdk.testing.types import ExampleData, RunInfo


def _get_data_directory() -> Path:
    """Get the path to the bundled data directory."""
    with resources.as_file(resources.files("chap_python_sdk") / "data") as data_path:
        return Path(data_path)


def _load_csv(file_path: Path) -> DataFrame:
    """Load a CSV file into a DataFrame."""
    return DataFrame.from_csv(file_path)


def list_available_datasets() -> list[tuple[str, str]]:
    """List all available example datasets as (country, frequency) tuples.

    Returns:
        List of (country, frequency) tuples for available datasets.
    """
    data_directory = _get_data_directory()
    datasets: list[tuple[str, str]] = []

    if not data_directory.exists():
        return datasets

    for example_directory in data_directory.iterdir():
        if not example_directory.is_dir():
            continue

        for frequency_directory in example_directory.iterdir():
            if not frequency_directory.is_dir():
                continue

            # Check if required files exist
            required_files = ["training_data.csv", "historic_data.csv", "future_data.csv"]
            if all((frequency_directory / f).exists() for f in required_files):
                # Extract country from directory name (e.g., "ewars_example" -> "laos")
                country = _extract_country_from_directory(example_directory.name)
                frequency = _normalize_frequency(frequency_directory.name)
                datasets.append((country, frequency))

    return datasets


def _extract_country_from_directory(directory_name: str) -> str:
    """Extract country name from directory name."""
    # Map directory names to country names
    directory_to_country = {
        "ewars_example": "laos",
    }
    return directory_to_country.get(directory_name, directory_name)


def _normalize_frequency(frequency: str) -> str:
    """Normalize frequency string."""
    frequency_map = {
        "monthly": "monthly",
        "M": "monthly",
        "weekly": "weekly",
        "W": "weekly",
        "daily": "daily",
        "D": "daily",
    }
    return frequency_map.get(frequency, frequency)


def _get_dataset_path(country: str, frequency: str) -> Path:
    """Get the path to a dataset directory."""
    data_directory = _get_data_directory()

    # Map country names to directory names
    country_to_directory = {
        "laos": "ewars_example",
    }

    # Map frequency to directory name
    frequency_to_directory = {
        "monthly": "monthly",
        "weekly": "weekly",
        "daily": "daily",
    }

    directory_name = country_to_directory.get(country.lower(), country)
    frequency_directory = frequency_to_directory.get(frequency.lower(), frequency)

    return data_directory / directory_name / frequency_directory


def get_example_data(
    country: str,
    frequency: str,
    configuration: dict[str, Any] | None = None,
) -> ExampleData:
    """Load example dataset for the specified country and frequency.

    Args:
        country: Country identifier (e.g., "laos").
        frequency: Data frequency (e.g., "monthly").
        configuration: Optional model configuration dictionary.

    Returns:
        ExampleData containing training_data, historic_data, future_data,
        and optionally predictions and configuration.

    Raises:
        FileNotFoundError: If the dataset does not exist.
    """
    dataset_path = _get_dataset_path(country, frequency)

    if not dataset_path.exists():
        available = list_available_datasets()
        available_str = ", ".join(f"({c}, {f})" for c, f in available)
        raise FileNotFoundError(
            f"Dataset not found for country='{country}', frequency='{frequency}'. Available datasets: {available_str}"
        )

    training_data = _load_csv(dataset_path / "training_data.csv")
    historic_data = _load_csv(dataset_path / "historic_data.csv")
    future_data = _load_csv(dataset_path / "future_data.csv")

    predictions: DataFrame | None = None
    predictions_path = dataset_path / "predictions.csv"
    if predictions_path.exists():
        predictions = _load_csv(predictions_path)

    # Create default RunInfo based on future_data
    run_info = RunInfo(prediction_length=len(future_data))

    return ExampleData(
        training_data=training_data,
        historic_data=historic_data,
        future_data=future_data,
        predictions=predictions,
        run_info=run_info,
        configuration=configuration,
        geo=None,
    )
