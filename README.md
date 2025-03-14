# CO2_Emission

This repository contains the code for my master's thesis project on CO2 emissions. The focus is on data processing, analysis, and modeling. Note that this repo includes only code with minimal documentation.

## Predictive Models

This repository contains predictive models for forecasting CO2 emissions. The implemented models include:
- **ARIMAX:** A statistical model that extends ARIMA by incorporating exogenous variables.
- **ARIMAX PCA:** A variation of ARIMAX that applies PCA for dimensionality reduction on exogenous inputs.
- **LightGBM:** A gradient boosting framework using tree-based learning for regression tasks.
- **LSTM:** A recurrent neural network model designed to capture long-term dependencies in time series data.
- **ARIMAX + LSTM:** A hybrid approach that combines ARIMAX and LSTM to leverage both statistical and deep learning methods.
- **ARIMAX PCA + LSTM:** A hybrid model that integrates PCA-enhanced ARIMAX with LSTM for improved forecasting.
- **LightGBM + LSTM:** An ensemble method that combines LightGBM and LSTM to enhance prediction accuracy.


## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/CO2_Emission.git
   cd CO2_Emission

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt

3. **Run the Code:**

- The main scripts are in the `/src` directory.
- Jupyter notebooks for exploratory analysis are in the `/notebooks` directory.