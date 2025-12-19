"""Type definitions for the testing module."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame
from geojson_pydantic import Feature, FeatureCollection
from pydantic import BaseModel, Field

GeoFeatureCollection = FeatureCollection[Feature[Any, Any]]


class RunInfo(BaseModel):
    """Runtime information passed from CHAP to models."""

    prediction_length: int = Field(description="Number of periods to predict")
    additional_continuous_covariates: list[str] = Field(
        default_factory=list,
        description="User-specified additional covariates present in the data",
    )
    future_covariate_origin: str | None = Field(
        default=None,
        description="Origin/source of future covariate forecasts",
    )


# Type aliases for functional model runner interface
type TrainFunction = Callable[
    [BaseConfig, DataFrame, RunInfo, GeoFeatureCollection | None],
    Awaitable[Any],
]
type PredictFunction = Callable[
    [BaseConfig, Any, DataFrame, DataFrame, RunInfo, GeoFeatureCollection | None],
    Awaitable[DataFrame],
]


@dataclass
class ExampleData:
    """Container for example dataset components."""

    training_data: DataFrame
    historic_data: DataFrame
    future_data: DataFrame
    predictions: DataFrame | None = None
    run_info: RunInfo | None = None
    configuration: dict[str, Any] | None = None
    geo: GeoFeatureCollection | None = None


def _empty_string_list() -> list[str]:
    """Return an empty list of strings."""
    return []


@dataclass
class ValidationResult:
    """Result of model I/O validation."""

    success: bool
    errors: list[str] = field(default_factory=_empty_string_list)
    warnings: list[str] = field(default_factory=_empty_string_list)
    n_predictions: int = 0
    n_samples: int = 0
