"""Type definitions for the testing module."""

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame
from geojson_pydantic import FeatureCollection

ConfigT = TypeVar("ConfigT", bound=BaseConfig)


@dataclass
class ExampleData:
    """Container for example dataset components."""

    training_data: DataFrame
    historic_data: DataFrame
    future_data: DataFrame
    predictions: DataFrame | None = None
    configuration: dict[str, Any] | None = None
    geo: FeatureCollection | None = None


@dataclass
class ValidationResult:
    """Result of model I/O validation."""

    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_predictions: int = 0
    n_samples: int = 0


class ModelRunnerProtocol(Protocol[ConfigT]):
    """Protocol for chapkit model runner interface."""

    async def on_init(self) -> None:
        """Optional initialization hook."""
        ...

    async def on_cleanup(self) -> None:
        """Optional cleanup hook."""
        ...

    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model and return the trained model object."""
        ...

    async def on_predict(
        self,
        config: ConfigT,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        """Make predictions using a trained model."""
        ...
