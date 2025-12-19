# Implementation Plan: Model Contract Alignment

This document outlines the changes needed across three repositories to align with the model contract defined in MODEL_CONTRACT.md.

## Summary of Contract Requirements

The MODEL_CONTRACT.md specifies:

1. **Run Info**: A separate `run_info` object passed to train/predict containing:
   - `prediction_length`: How many periods the model should predict
   - `additional_continuous_covariates`: User-specified extra covariates
   - `future_covariate_origin`: Origin of future covariate data

2. **Model Config**: Unchanged - model-specific user options via pydantic ModelConfig class

3. **Model Information**: Models expose their requirements via:
   - `required_covariates`: List of required input columns
   - `allow_free_additional_continuous_covariates`: Whether extra covariates are accepted

4. **Prediction Format**: Wide format with `sample_0`, `sample_1`, ... columns

---

## New Function Signatures

```python
# Train function
async def on_train(
    config: BaseConfig,           # Model-specific config (unchanged)
    data: DataFrame,              # Training data
    run_info: RunInfo,            # NEW: CHAP runtime info
    geo: GeoFeatureCollection | None = None,
) -> Any:

# Predict function
async def on_predict(
    config: BaseConfig,           # Model-specific config (unchanged)
    model: Any,                   # Trained model
    historic: DataFrame,          # Historic data
    future: DataFrame,            # Future data to predict
    run_info: RunInfo,            # NEW: CHAP runtime info
    geo: GeoFeatureCollection | None = None,
) -> DataFrame:
```

---

## Part A: chap-python-sdk Updates

### A.1 Add RunInfo Type (types.py)

```python
from pydantic import BaseModel, Field

class RunInfo(BaseModel):
    """Runtime information passed from CHAP to models."""

    prediction_length: int = Field(description="Number of periods to predict")
    additional_continuous_covariates: list[str] = Field(
        default_factory=list,
        description="User-specified additional covariates"
    )
    future_covariate_origin: str | None = Field(
        default=None,
        description="Origin/source of future covariate forecasts"
    )
```

### A.2 Update Function Type Aliases (types.py)

```python
# Updated function signatures with run_info parameter
type TrainFunction = Callable[
    [BaseConfig, DataFrame, RunInfo, GeoFeatureCollection | None],
    Awaitable[Any]
]

type PredictFunction = Callable[
    [BaseConfig, Any, DataFrame, DataFrame, RunInfo, GeoFeatureCollection | None],
    Awaitable[DataFrame]
]
```

### A.3 Update ExampleData (types.py)

```python
@dataclass
class ExampleData:
    """Container for example dataset components."""

    training_data: DataFrame
    historic_data: DataFrame
    future_data: DataFrame
    predictions: DataFrame | None = None
    run_info: RunInfo | None = None  # NEW
    configuration: dict[str, Any] | None = None
    geo: GeoFeatureCollection | None = None
```

### A.4 Update Validation (validation.py)

```python
async def validate_model_io(
    train_function: TrainFunction,
    predict_function: PredictFunction,
    example_data: ExampleData,
    config: BaseConfig | None = None,
    run_info: RunInfo | None = None,  # NEW parameter
) -> ValidationResult:
    """Validate model train/predict functions against example data."""

    if config is None:
        config = BaseConfig()

    # Use provided run_info or create default from example_data or defaults
    if run_info is None:
        run_info = example_data.run_info or RunInfo(
            prediction_length=len(example_data.future_data),
            additional_continuous_covariates=[],
        )

    runner = FunctionalModelRunner(on_train=train_function, on_predict=predict_function)

    # Train with run_info
    trained_model = await runner.on_train(
        config=config,
        data=example_data.training_data,
        run_info=run_info,
        geo=example_data.geo,
    )

    # Predict with run_info
    predictions = await runner.on_predict(
        config=config,
        model=trained_model,
        historic=example_data.historic_data,
        future=example_data.future_data,
        run_info=run_info,
        geo=example_data.geo,
    )
```

### A.5 Add Wide Format Validation (assertions.py)

```python
def assert_wide_format_predictions(predictions: DataFrame) -> None:
    """Assert predictions are in wide format (sample_0, sample_1, ...)."""
    sample_columns = [c for c in predictions.columns if c.startswith("sample_")]
    if not sample_columns:
        raise PredictionValidationError(
            "Predictions must have sample columns (sample_0, sample_1, ...)"
        )

def assert_nonnegative_predictions(predictions: DataFrame) -> None:
    """Assert all prediction values are non-negative."""
    sample_columns = [c for c in predictions.columns if c.startswith("sample_")]
    for col in sample_columns:
        values = predictions[col]
        for idx, val in enumerate(values):
            if val < 0:
                raise PredictionValidationError(
                    f"Row {idx}, {col}: Negative value {val} not allowed"
                )

def assert_no_nan_predictions(predictions: DataFrame) -> None:
    """Assert no NaN values in predictions."""
    sample_columns = [c for c in predictions.columns if c.startswith("sample_")]
    for col in sample_columns:
        values = predictions[col]
        for idx, val in enumerate(values):
            if val != val:  # NaN check
                raise PredictionValidationError(
                    f"Row {idx}, {col}: NaN value not allowed"
                )
```

### A.6 Update Example Data Loading (example_data.py)

Add default RunInfo to loaded example data:

```python
def get_example_data(
    country: str,
    frequency: str,
    configuration: dict[str, Any] | None = None,
) -> ExampleData:
    # ... existing loading code ...

    # Create default run_info based on future_data
    run_info = RunInfo(
        prediction_length=len(future_data),
        additional_continuous_covariates=[],
    )

    return ExampleData(
        training_data=training_data,
        historic_data=historic_data,
        future_data=future_data,
        predictions=predictions,
        run_info=run_info,  # NEW
        configuration=configuration,
        geo=None,
    )
```

### A.7 Files to Modify

| File | Changes |
|------|---------|
| `types.py` | Add RunInfo class; update TrainFunction/PredictFunction signatures; add run_info to ExampleData |
| `validation.py` | Add run_info parameter; pass to train/predict calls |
| `assertions.py` | Add assert_wide_format_predictions, assert_nonnegative_predictions, assert_no_nan_predictions |
| `example_data.py` | Create default RunInfo when loading example data |
| `__init__.py` | Export RunInfo |

---

## Part B: chapkit Updates (PR)

### B.1 Add RunInfo Type (config/schemas.py)

```python
class RunInfo(BaseModel):
    """Runtime information passed from CHAP to models."""

    prediction_length: int
    additional_continuous_covariates: list[str] = Field(default_factory=list)
    future_covariate_origin: str | None = None
```

### B.2 Update Runner Signatures (ml/runner.py)

```python
type TrainFunction[ConfigT] = Callable[
    [ConfigT, DataFrame, RunInfo, FeatureCollection | None],
    Awaitable[Any]
]

type PredictFunction[ConfigT] = Callable[
    [ConfigT, Any, DataFrame, DataFrame, RunInfo, FeatureCollection | None],
    Awaitable[DataFrame]
]


class FunctionalModelRunner(BaseModelRunner[ConfigT]):
    """Functional model runner wrapping train and predict functions."""

    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        run_info: RunInfo,  # NEW
        geo: FeatureCollection | None = None,
    ) -> Any:
        return await self._on_train(config, data, run_info, geo)

    async def on_predict(
        self,
        config: ConfigT,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        run_info: RunInfo,  # NEW
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        return await self._on_predict(config, model, historic, future, run_info, geo)
```

### B.3 Update BaseModelRunner (ml/runner.py)

```python
class BaseModelRunner(ABC, Generic[ConfigT]):
    """Abstract base class for model runners."""

    @abstractmethod
    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        run_info: RunInfo,  # NEW
        geo: FeatureCollection | None = None,
    ) -> Any:
        ...

    @abstractmethod
    async def on_predict(
        self,
        config: ConfigT,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        run_info: RunInfo,  # NEW
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        ...
```

### B.4 Update Request Schemas (ml/schemas.py)

```python
class TrainRequest(BaseModel):
    """Request to train a model."""

    config_id: ULID
    data: DataFrame
    geo: FeatureCollection | None = None
    # NEW: Run info fields
    prediction_length: int
    additional_continuous_covariates: list[str] = Field(default_factory=list)
    future_covariate_origin: str | None = None


class PredictRequest(BaseModel):
    """Request to make predictions."""

    training_artifact_id: ULID
    historic: DataFrame
    future: DataFrame
    geo: FeatureCollection | None = None
    # NEW: Run info fields (may differ from training)
    prediction_length: int | None = None  # If None, infer from future data
    additional_continuous_covariates: list[str] = Field(default_factory=list)
    future_covariate_origin: str | None = None
```

### B.5 Update MLManager (ml/manager.py)

```python
async def _train_task(self, request: TrainRequest, artifact_id: ULID) -> ULID:
    config = await config_manager.find_by_id(request.config_id)

    # Build RunInfo from request
    run_info = RunInfo(
        prediction_length=request.prediction_length,
        additional_continuous_covariates=request.additional_continuous_covariates,
        future_covariate_origin=request.future_covariate_origin,
    )

    training_result = await self.runner.on_train(
        config=config.data,
        data=request.data,
        run_info=run_info,  # NEW
        geo=request.geo,
    )


async def _predict_task(self, request: PredictRequest, artifact_id: ULID) -> ULID:
    # ... load training artifact and config ...

    # Build RunInfo from request
    run_info = RunInfo(
        prediction_length=request.prediction_length or len(request.future),
        additional_continuous_covariates=request.additional_continuous_covariates,
        future_covariate_origin=request.future_covariate_origin,
    )

    predictions = await self.runner.on_predict(
        config=config.data,
        model=trained_model,
        historic=request.historic,
        future=request.future,
        run_info=run_info,  # NEW
        geo=request.geo,
    )
```

### B.6 Files to Modify

| File | Changes |
|------|---------|
| `config/schemas.py` | Add RunInfo class |
| `ml/runner.py` | Add run_info parameter to all runner methods and type aliases |
| `ml/schemas.py` | Add run info fields to TrainRequest/PredictRequest |
| `ml/manager.py` | Build RunInfo and pass to runner |
| `ml/__init__.py` | Export RunInfo |

---

## Part C: chap_core Updates (PR)

### C.1 Update REST API Wrapper (models/chapkit_rest_api_wrapper.py)

```python
def train(
    self,
    config_id: str,
    data: pd.DataFrame,
    geo: dict | None = None,
    prediction_length: int = 3,
    additional_continuous_covariates: list[str] | None = None,
    future_covariate_origin: str | None = None,
) -> dict:
    """Submit training job with run info."""

    payload = {
        "config_id": config_id,
        "data": {"columns": list(data.columns), "data": data.values.tolist()},
        "geo": geo,
        "prediction_length": prediction_length,
        "additional_continuous_covariates": additional_continuous_covariates or [],
        "future_covariate_origin": future_covariate_origin,
    }

    response = self._post("/api/v1/ml/train", json=payload)
    return response.json()


def predict(
    self,
    artifact_id: str,
    historic_data: pd.DataFrame,
    future_data: pd.DataFrame,
    geo: dict | None = None,
    prediction_length: int | None = None,
    additional_continuous_covariates: list[str] | None = None,
    future_covariate_origin: str | None = None,
) -> dict:
    """Submit prediction job with run info."""

    payload = {
        "training_artifact_id": artifact_id,
        "historic": {"columns": list(historic_data.columns), "data": historic_data.values.tolist()},
        "future": {"columns": list(future_data.columns), "data": future_data.values.tolist()},
        "geo": geo,
        "prediction_length": prediction_length or len(future_data),
        "additional_continuous_covariates": additional_continuous_covariates or [],
        "future_covariate_origin": future_covariate_origin,
    }

    response = self._post("/api/v1/ml/predict", json=payload)
    return response.json()
```

### C.2 Update ExternalChapkitModel (models/external_chapkit_model.py)

```python
def train(self, train_data: DataSet, extra_args=None):
    frequency = self._get_frequency(train_data)
    df = train_data.to_pandas()
    new_df = self._adapt_data(df, frequency=frequency)
    geo = train_data.polygons

    # Get run info from model configuration
    prediction_length = getattr(self._model_configuration, 'prediction_length', 3)
    additional_covariates = getattr(
        self._model_configuration, 'additional_continuous_covariates', []
    )
    future_covariate_origin = getattr(
        self._model_configuration, 'future_covariate_origin', None
    )

    response = self.client.train_and_wait(
        self.configuration_id,
        new_df,
        geo,
        prediction_length=prediction_length,
        additional_continuous_covariates=additional_covariates,
        future_covariate_origin=future_covariate_origin,
    )
    self._train_id = response["artifact_id"]


def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
    # ... existing code ...

    response = self.client.predict_and_wait(
        artifact_id=self._train_id,
        future_data=future_data_pd,
        historic_data=historic_data_pd,
        geo_features=geo,
        prediction_length=len(future_data),
        additional_continuous_covariates=self._additional_covariates,
        future_covariate_origin=self._future_covariate_origin,
    )
```

### C.3 Update ModelConfiguration (database/model_templates_and_config_tables.py)

```python
class ModelConfiguration(SQLModel):
    """Configuration for a specific model instance."""

    user_option_values: Optional[dict] = {}
    additional_continuous_covariates: List[str] = []
    prediction_length: int = 3  # NEW
    future_covariate_origin: Optional[str] = None  # NEW
```

### C.4 Files to Modify

| File | Changes |
|------|---------|
| `models/chapkit_rest_api_wrapper.py` | Add run info parameters to train/predict methods |
| `models/external_chapkit_model.py` | Pass run info from config to API calls |
| `database/model_templates_and_config_tables.py` | Add prediction_length, future_covariate_origin fields |

---

## Migration Guide for Model Authors

### Before (current)
```python
async def on_train(
    config: BaseConfig,
    data: DataFrame,
    geo: GeoFeatureCollection | None = None,
) -> Any:
    learning_rate = config.learning_rate
    return train_model(data, learning_rate)


async def on_predict(
    config: BaseConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: GeoFeatureCollection | None = None,
) -> DataFrame:
    return model.predict(future)
```

### After (new contract)
```python
async def on_train(
    config: BaseConfig,
    data: DataFrame,
    run_info: RunInfo,  # NEW parameter
    geo: GeoFeatureCollection | None = None,
) -> Any:
    learning_rate = config.learning_rate  # Config unchanged
    prediction_length = run_info.prediction_length  # Access via run_info
    return train_model(data, learning_rate, prediction_length)


async def on_predict(
    config: BaseConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    run_info: RunInfo,  # NEW parameter
    geo: GeoFeatureCollection | None = None,
) -> DataFrame:
    # Can use run_info.additional_continuous_covariates if needed
    return model.predict(future)
```

---

## Prediction Output Format

### Required Format (Wide)
```
time_period | location | sample_0 | sample_1 | sample_2 | ...
2024-01     | Bokeo    | 15.2     | 18.1     | 12.9     | ...
2024-02     | Bokeo    | 22.1     | 19.5     | 25.3     | ...
```

### Validation Rules
- All `sample_*` columns must contain numeric values
- All values must be non-negative (>= 0)
- No NaN values allowed
- Must have `time_period` and `location` columns
- Row count must match `future_data` row count

---

## Implementation Order

### Phase 1: chap-python-sdk (This Repo)
1. Add RunInfo class to types.py
2. Update TrainFunction/PredictFunction signatures
3. Add run_info to ExampleData
4. Update validation.py to pass run_info
5. Add wide format assertions
6. Update example_data.py to create default RunInfo
7. Update all tests
8. Commit and push

### Phase 2: chapkit (PR)
1. Add RunInfo to config/schemas.py
2. Update BaseModelRunner abstract methods
3. Update FunctionalModelRunner
4. Update type aliases
5. Update request schemas
6. Update MLManager
7. Update tests
8. Create PR

### Phase 3: chap_core (PR)
1. Update REST API wrapper
2. Update ExternalChapkitModel
3. Update ModelConfiguration schema
4. Update tests
5. Create PR
