# 📈 ConformL - Time Series Forecasting Tool

> **A user-friendly GUI application for time series forecasting with uncertainty quantification**

ConformL is an easy-to-use forecasting tool that helps you predict future values in your time series data while showing you how confident those predictions are. No coding required - just upload your data, configure your settings, and get professional-grade forecasts with confidence intervals.

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## ✨ What Makes ConformL Special?

- **🎯 No Coding Required** - Easy-to-use web interface, perfect for business analysts and non-technical users
- **📊 Multiple Forecasting Models** - Choose from ARIMA, SARIMA, Prophet, and XGBoost
- **🔮 Uncertainty Quantification** - Not just predictions, but confidence intervals that tell you the range of likely outcomes
- **📈 Interactive Visualizations** - Beautiful, interactive charts powered by Plotly
- **🏆 Model Comparison** - Run multiple models simultaneously and see which performs best
- **💾 Export Everything** - Download predictions as CSV and charts as high-quality PNG images
- **⚡ Real-time Updates** - See results instantly as you adjust parameters

---

## 🎬 Quick Start

### Installation

1. **Install Python 3.8 or higher** (if not already installed)
   - Download from [python.org](https://www.python.org/downloads/)

2. **Clone or download this repository**
   ```bash
   git clone https://github.com/Jnaneshp/Conforml-package.git
   cd build
   ```

3. **Install dependencies**
   ```bash
   pip install -r conforml/requirements.txt
   pip install -e .
   ```

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to that address

---

## 🎯 How to Use ConformL

### Step 1: Load Your Data

Two options:

**Option A: Try the Sample Dataset**
- Select "Sample Dataset (Delhi Temperature)" from the sidebar
- This loads historical temperature data for you to explore

**Option B: Upload Your Own CSV File**
- Select "Upload CSV File"
- Your CSV must have:
  - **First column**: Timestamps (dates or datetime)
  - **Second column**: Values (numbers you want to forecast)

Example CSV format:
```csv
Date,Value
2024-01-01,25.3
2024-01-02,26.1
2024-01-03,24.8
...
```

### Step 2: Configure Your Forecast

**1. Choose a Forecasting Model**

In the sidebar under "🤖 Model Selection":

- **ARIMA** - Best for stable trends and patterns
  - Adjust p (autoregressive order), d (differencing), q (moving average)
  
- **SARIMA** - Best when your data has seasonal patterns (e.g., monthly sales)
  - Same as ARIMA but with seasonal parameters
  
- **Prophet** - Best for daily data with strong seasonal effects
  - Developed by Facebook, great for business metrics
  
- **XGBoost** - Best for complex, non-linear patterns
  - Machine learning approach, handles many patterns

**2. Choose Conformal Prediction Method**

This determines how we create confidence intervals:

- **CVPlusConformal** - Best for stable, consistent data
  - Adjust "CV Folds" (more folds = more robust, slower)
  
- **AdaptiveConformal** - Best for changing/evolving patterns
  - Choose "decay" for gradual changes or "sliding" for sudden shifts
  
- **ISOC** - Balances accuracy with tighter confidence intervals
  - Adjust "Sharpness Weight" (higher = tighter intervals)

**3. Set Confidence Level**

- **Significance Level (α)**: Lower values = wider confidence intervals
  - α = 0.05 gives you 95% confidence intervals
  - α = 0.10 gives you 90% confidence intervals

**4. Configure Data Split**

- **Test Steps**: How many future points to forecast (default: 50)

### Step 3: Run Your Forecast

**Single Model Mode:**
1. Click "🚀 Run Forecast (single model)"
2. Wait for the model to train (10 seconds to 2 minutes depending on model)
3. View your results!

**Model Comparison Mode:**
1. Select multiple models in the "🔁 Model Compare" section
2. Click "▶️ Run Compare (selected models)"
3. See leaderboard showing which model performs best
4. Inspect individual model results

---

## 📊 Understanding Your Results

### Main Forecast Chart

- **Black line**: Actual historical values
- **Orange line**: Model predictions
- **Red shaded area**: Confidence interval (range where future values likely fall)

### Performance Metrics

- **RMSE (Root Mean Squared Error)**: Average prediction error (lower is better)
- **MAPE (Mean Absolute Percentage Error)**: Error as a percentage (lower is better)

### Conformal Prediction Metrics

- **Target Coverage**: What percentage should fall in the interval (based on your α setting)
- **Empirical Coverage**: What percentage actually fell in the interval
  - ✅ Green: Good match between target and empirical
  - ⚠️ Yellow: Acceptable difference
  - ❌ Red: Poor coverage, adjust settings
- **Avg Interval Width**: How wide your confidence intervals are

### Additional Charts

1. **Coverage Indicator**: Shows which points fell inside/outside confidence intervals
2. **Interval Width Over Time**: Shows if uncertainty changes over time
3. **Absolute Error**: Shows prediction accuracy at each point

---

## 💡 Tips for Best Results

### Choosing the Right Model

**Use ARIMA when:**
- Your data shows a clear trend or pattern
- Data is relatively stable over time
- You want fast, interpretable results

**Use SARIMA when:**
- Your data has seasonal patterns (e.g., monthly, quarterly cycles)
- Same day/week/month typically has similar values
- Example: Retail sales, weather data

**Use Prophet when:**
- You have daily data with multiple seasonal patterns
- Data has holidays or special events
- You want automatic handling of missing data

**Use XGBoost when:**
- Your data has complex, non-linear patterns
- Traditional models aren't performing well
- You have enough historical data (200+ points)

### Choosing the Right Conformal Method

**Use CVPlusConformal when:**
- Your data patterns are consistent over time
- You have enough data for cross-validation (500+ points)
- You want robust, reliable intervals

**Use AdaptiveConformal when:**
- Your data characteristics change over time
- Recent patterns are more important than old ones
- Choose "decay" for smooth transitions, "sliding" for abrupt changes

**Use ISOC when:**
- You want tighter confidence intervals
- You're willing to trade some coverage for precision
- Use in low-risk decision scenarios

### Tuning Parameters

**If your intervals are too wide:**
- Increase α (e.g., from 0.05 to 0.10)
- Use ISOC with higher sharpness weight
- Try AdaptiveConformal with shorter window

**If coverage is too low (missing actual values):**
- Decrease α (e.g., from 0.10 to 0.05)
- Increase CV folds in CVPlus
- Use a different model that fits your data better

**If predictions are inaccurate:**
- Try different model types
- Adjust model parameters
- Check if you have enough training data
- Look for outliers or data quality issues

---

## 📁 Exporting Your Results

### Download Predictions
- Click "💾 Download Results" button
- Saves CSV file with: timestamp, actual, predicted, lower bound, upper bound
- Import into Excel or other tools for further analysis

### Download Charts
- Click any "💾 Download ... Plot (PNG)" button
- High-quality PNG images (2x resolution)
- Perfect for reports and presentations

**Available exports:**
- Forecast plot with intervals
- Coverage indicator chart
- Interval width evolution
- Absolute error chart

---

## 🏆 Model Comparison Feature

Compare multiple models to find the best one for your data:

1. **Select models** in the sidebar under "🔁 Model Compare"
2. **Click "Run Compare"** - all selected models run automatically
3. **View leaderboard** - sorted by RMSE (best performer at top)
4. **Inspect results** - click on any model to see detailed charts
5. **Choose winner** - use the best-performing model for your forecasts

The comparison runs all models with the same settings, making it a fair apples-to-apples comparison.

---

## 🎓 What is Conformal Prediction?

### In Simple Terms

Traditional forecasting tools give you a single prediction: "Next month's sales will be $50,000."

ConformL gives you: "Next month's sales will be $50,000, and we're 95% confident it will be between $45,000 and $55,000."

This **confidence interval** helps you:
- Plan for best and worst-case scenarios
- Know when predictions are uncertain
- Make better business decisions

### Why It Matters

- **No assumptions needed**: Works with any data distribution
- **Mathematically guaranteed**: Coverage levels are proven, not guessed
- **Adapts to your data**: Intervals get wider when data is unpredictable

---

## 📈 Use Cases

### Business Applications

**Sales Forecasting**
- Predict future revenue with confidence intervals for budgeting
- Model: Prophet (handles weekly/monthly patterns)
- Method: CVPlus

**Inventory Management**
- Forecast demand to optimize stock levels
- Model: SARIMA (seasonal patterns)
- Method: Adaptive (demand changes over time)

**Financial Planning**
- Project cash flow with uncertainty quantification
- Model: XGBoost (complex patterns)
- Method: ISOC (tighter intervals for planning)

### Technical Applications

**Server Load Prediction**
- Forecast traffic for capacity planning
- Model: Prophet (daily/weekly patterns)
- Method: Adaptive Sliding (recent patterns most important)

**Equipment Monitoring**
- Predict sensor readings for maintenance
- Model: ARIMA (stable trends)
- Method: CVPlus (consistent patterns)

**Energy Consumption**
- Forecast usage for grid management
- Model: SARIMA (strong seasonality)
- Method: Adaptive Decay (gradual pattern changes)

---

## 🛠️ Technical Details

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for installation

### Dependencies

Core packages:
- Streamlit (web interface)
- Plotly (interactive charts)
- Scikit-learn (machine learning)
- Statsmodels (statistical models)
- Prophet (Facebook forecasting)
- XGBoost (gradient boosting)

See `conforml/requirements.txt` for complete list.

---

## 🐛 Troubleshooting

### App won't start

**Problem**: `ModuleNotFoundError`
- **Solution**: Run `pip install -r conforml/requirements.txt` again

**Problem**: Port already in use
- **Solution**: Run `streamlit run app.py --server.port 8502`

### Forecast errors

**Problem**: "Model must be fitted before making predictions"
- **Solution**: Click "Run Forecast" button again

**Problem**: Prophet errors on Windows
- **Solution**: Install Prophet separately: `pip install prophet`

**Problem**: Out of memory error
- **Solution**: Reduce test steps or use smaller dataset

### Poor results

**Problem**: Low coverage (actual values outside intervals)
- **Solution**: Decrease α or try different conformal method

**Problem**: Very wide intervals
- **Solution**: Increase α, try ISOC, or use AdaptiveConformal

**Problem**: High RMSE/MAPE
- **Solution**: Try different model types, check data quality, adjust model parameters

---

## 📚 Additional Resources

### Sample Datasets

ConformL includes sample data to help you get started:
- **Delhi Temperature**: Daily minimum temperatures (1981-1990)
- Good for: Learning SARIMA and seasonal patterns

### Documentation

- **README.md** (this file): User guide
- **doc.md**: Technical documentation with algorithms and theory
- **conforml/README.md**: Python API documentation (for developers)

### Support

Found a bug or have a question?
- Check the troubleshooting section above
- Review documentation in `doc.md`
- Open an issue on GitHub

---

## 🎨 Screenshots

*[Add screenshots of your app here showing:]*
1. Main interface with data loaded
2. Forecast visualization with confidence intervals
3. Model comparison leaderboard
4. Conformal metrics charts

---

## 🔄 Updates and Roadmap

### Current Version: 0.1.0

**Features:**
✅ Four forecasting models  
✅ Three conformal prediction methods  
✅ Interactive Streamlit interface  
✅ Model comparison mode  
✅ CSV and PNG exports  
✅ Real-time parameter tuning  

**Coming Soon:**
- [ ] More forecasting models (LSTM, GRU)
- [ ] Automated model selection
- [ ] Batch processing for multiple files
- [ ] PDF report generation
- [ ] API endpoint for integration

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Conformal prediction based on research by Vovk et al.
- Prophet model by Facebook
- Sample data from Australian Bureau of Meteorology

---

## 🚀 Getting Started Checklist

- [ ] Install Python 3.8+
- [ ] Clone/download repository
- [ ] Install dependencies (`pip install -r conforml/requirements.txt`)
- [ ] Run app (`streamlit run app.py`)
- [ ] Try sample dataset
- [ ] Upload your own data
- [ ] Run first forecast
- [ ] Compare multiple models
- [ ] Export results

**Ready to predict the future? Launch the app and start forecasting! 🎯**
