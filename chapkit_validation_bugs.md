# Chapkit Model Validation Test Cases

Bugs discovered during chap_pymc chapkit integration testing with `evaluate2`.

## 1. Type Mismatch in Config Parameters

**Bug**: Config parameter type didn't match underlying model's expected type.

**Example**:
- Chapkit config had `mixture_weight_prior: float = 0.5`
- Model expected `mixture_weight_prior: tuple[float, float] = (0.5, 0.5)`

**Error**:
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for FourierHyperparameters
mixture_weight_prior
  Input should be a valid tuple [type=tuple_type, input_value=0.5, input_type=float]
```

**Validation Test**:
- Verify that types exposed in chapkit `BaseConfig` subclass can be correctly passed to underlying model's Pydantic models
- Test that `on_predict` can be called with default config values without type errors
- Schema validation: ensure config schema types are compatible with model internals

---

## 2. Missing Required Interface Properties (chap-core issue)

**Bug**: `ExternalChapkitModelTemplate` in chap-core accessed `.model_template_config` as an attribute, but the class only had `get_model_template_config()` method.

**Error**:
```
AttributeError: 'ExternalChapkitModelTemplate' object has no attribute 'model_template_config'.
Did you mean: 'get_model_template_config'?
```

**Fix Applied**: Added `@property` wrapper in chap-core:
```python
@property
def model_template_config(self) -> ModelTemplateConfigV2:
    return self.get_model_template_config()
```

**Validation Test**:
- Test that chapkit model templates expose required interface attributes/properties for compatibility with chap-core evaluation tools
- Integration test: run a minimal backtest against a chapkit model to verify full pipeline compatibility

---

## 3. None Value for Required String Field (chap-core issue)

**Bug**: `author_note` from chapkit info endpoint was `None`, but `ModelTemplateConfigV2` required a non-null string.

**Error**:
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelTemplateConfigV2
meta_data.author_note
  Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]
```

**Fix Applied**: Changed in chap-core:
```python
# Before
"author_note": model_info.get("author_note", ""),
# After
"author_note": model_info.get("author_note") or "",
```

**Validation Test**:
- Test that chapkit's `/api/v1/info` endpoint returns all required fields with correct types
- Test that optional fields (that may be None) are handled gracefully by consumers
- Verify `MLServiceInfo` fields map correctly to what chap-core expects

---

## 4. Duplicate Parameter Definitions Not Synchronized

**Bug**: Model had duplicate `lag` parameters (constructor param and `Params.lag`) that were not synchronized, causing assertion failures.

**Example**:
```python
class Params(pydantic.BaseModel):
    lag: int = 3  # Default in Params

def __init__(self, lag: int = 3, params: Params = Params()):
    self._lag = lag  # Uses constructor default (3)
    self._params = params  # params.lag could be 1

# Later:
X = X.isel(epi_offset=slice(-self._lag, None))  # Uses self._lag (3)
assert X.shape[-1] == self._params.lag  # Checks self._params.lag (1) - FAILS!
```

**Error**:
```
AssertionError  # X.shape[-1] == 3, but self._params.lag == 1
```

**Validation Test**:
- Test that model config parameters actually affect model behavior
- Test with non-default config values to catch hardcoded defaults
- Verify config values propagate through the entire model pipeline

---

## Recommended Validation Test Suite

### 1. Config Type Validation
```python
def test_config_types_match_model_expectations():
    """Verify config field types are compatible with model internals."""
    config = MyModelConfig()
    # Should not raise type errors when passed to model
    runner.on_predict(config, model, historic, future)
```

### 2. Default Config Smoke Test
```python
def test_predict_with_default_config():
    """Verify model works with default config values."""
    config = MyModelConfig()
    result = await runner.on_predict(config, None, historic_df, future_df)
    assert result is not None
```

### 3. Non-Default Config Test
```python
def test_predict_with_custom_config():
    """Verify custom config values actually affect model behavior."""
    config = MyModelConfig(lag=1, n_harmonics=4)  # Non-defaults
    result = await runner.on_predict(config, None, historic_df, future_df)
    # Should not fail due to hardcoded defaults
```

### 4. Info Endpoint Completeness
```python
def test_info_endpoint_fields():
    """Verify info endpoint returns required fields."""
    response = client.get("/api/v1/info")
    info = response.json()
    assert info.get("display_name") is not None
    assert info.get("version") is not None
    # Check optional fields don't break consumers when None
```

### 5. End-to-End Integration Test
```python
def test_evaluate2_compatibility():
    """Verify model works with chap-core evaluate2 pipeline."""
    # Run minimal backtest to catch interface mismatches
    result = run_evaluate2(model_url, dataset, n_splits=1)
    assert result.status == "success"
```