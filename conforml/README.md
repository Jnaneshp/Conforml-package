rf# Conforml - Conformal Prediction Package

Conforml is a Python package that combines time series forecasting with conformal prediction methods to generate reliable prediction intervals. It provides implementations of various time series models and conformal prediction algorithms, making it easy to quantify uncertainty in time series forecasts.

## Features

- **Model-agnostic architecture**: Use any forecasting model implementing the base interface
- Multiple time series models:
  - Prophet for automated forecasting
  - LSTM and GRU neural networks for complex patterns
- Supported Conformal Methods:
  - Split Conformal
  - CV+ (Cross Validation+)
  - Adaptive Conformal

Model Integration Options:

- ARIMA
- LSTM
- GRU
- Prophet
- Custom Model Support
- Evaluation Metrics:
  - Coverage Rate
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - Interval Width
  - Sharpness
  - Pinball Loss
  - Calibration Error
- Easy-to-use data loading and preprocessing utilities

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Using Built-in Models

```python
# Advanced time series splitting for temporal data
from conforml.conformal import CVPlusConformal, AdaptiveConformal

# CV+ for stationary series
cv_predictor = CVPlusConformal(model, n_folds=5, alpha=0.1)
cv_predictor.fit(X_train, y_train)

# Adaptive conformal for non-stationary series
adaptive_predictor = AdaptiveConformal(model, threshold=0.1, alpha=0.1)
adaptive_predictor.fit(X_rolling_window)
```

### Using Built-in Models

```python
from conforml.data import TimeSeriesLoader, TimeSeriesPreprocessor
from conforml.models import ProphetModel
from conforml.conformal import SplitConformal

# Load and preprocess data
loader = TimeSeriesLoader()
loader.load_from_csv('data.csv', 'timestamp', 'value')
X, y = loader.get_data()

# Create and fit model with conformal prediction
model = ProphetModel()
conformal_predictor = SplitConformal(model, alpha=0.1)
conformal_predictor.fit(X, y)

# Make predictions with uncertainty intervals
predictions, lower, upper = conformal_predictor.predict(X_test)
```

## Advanced Features

### Temporal Cross-Validation

```python
from conforml.utils import temporal_train_test_split

# Preserve temporal order while splitting
X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=0.2)
```

### Online Updates

```python
# Update model and conformal intervals incrementally
adaptive_predictor.update(new_X, new_y)
```

## Custom Model Implementation

```python
from conforml.models.base import TimeSeriesModel
from conforml.conformal import SplitConformal

class CustomModel(TimeSeriesModel):
    def __init__(self):
        super().__init__()
        # Add your model here

    def fit(self, X, y=None):
        # Implement training
        return self

    def predict(self, X):
        # Return predictions
        return np.zeros_like(X)

# Usage with conformal prediction
model = CustomModel()
conformal = SplitConformal(model=model, alpha=0.1)
```

## Documentation

Detailed documentation is available in the `docs` directory. Build the documentation using Sphinx:

```bash
cd docs
make html
```

## Examples

Check the `examples` directory for practical usage examples:

- `streamflow_example.py`: Time series forecasting for streamflow data
- `evaluation_example.py`: Evaluating prediction intervals
- Key Visualization Features:
  - Time Series Predictions with Uncertainty Intervals
  - Rolling Coverage Rate Evolution
  - Model/Method Performance Comparisons

Example Usage (from evaluation_example.py):

```python
# After calculating metrics
rmse_calc = RMSECalculator()
mape_calc = MAPECalculator()
rmse_calc.update(y_test_orig, predictions)
mape_calc.update(y_test_orig, predictions)

print(f'RMSE: {rmse_calc.get_rmse():.2f}')
print(f'MAPE: {mape_calc.get_mape():.2f}%')
plot_time_series_intervals(timestamps, y_test_orig, predictions, lower, upper)
plot_coverage_evolution(coverage_rates)
plt.show()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
