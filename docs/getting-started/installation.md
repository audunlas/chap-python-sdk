# Installation

## Requirements

- Python 3.13 or higher
- [chapkit](https://github.com/climateview/chapkit) - Model framework

## Using uv (Recommended)

```bash
uv add chap-python-sdk
```

## Using pip

```bash
pip install chap-python-sdk
```

## Development Installation

To install the SDK for development:

```bash
git clone https://github.com/knutdrand/chap-python-sdk.git
cd chap-python-sdk
uv sync
```

### Running Tests

```bash
make test
```

### Linting

```bash
make lint
```

## Verifying Installation

After installation, verify that everything works:

```python
from chap_python_sdk.testing import list_available_datasets

datasets = list_available_datasets()
print(datasets)  # [("laos", "monthly")]
```
