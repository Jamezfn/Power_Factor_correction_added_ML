# Predicting Energy Consumption using LSTM

This project implements an LSTM-based Machine Learning model to predict the Energy consumption in kWh. The pipeline includes data preprocessing, model training, and evaluation, with a modularized structure for flexible updates and testing.

## Project Structure

```
BMS/
├── __init__.py
├── .gitignore
├── config.py       # Configuration parameters
├── README.md       # This file
├── requirements.txt # Python dependencies
├── data/           # Processed data, saved models, and scalers
├── env/            # Python virtual environment
├── model/          # LSTM model architecture
├── plots/          # Evaluation plots
├── rawData/        # Raw CSV data files
├── scripts/        # Preprocessing, training, and evaluation scripts
└── tests/          # Unit tests
```

## Setup

### 1. Clone the Repository
Open PowerShell and run:

```powershell
git clone https://github.com/fawannjoroge/BMS
cd BMS
```

### 2. Create and Activate the Virtual Environment
Create a virtual environment and activate it:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

> **Note:** If you encounter an execution policy error, run:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

## Running the Project

### Data Preprocessing
The script loads raw CSV data, cleans and scales it, generates sequences for the LSTM model, and saves processed data.

Run the preprocessing script:

```powershell
python -m scripts.preprocessing
```

### Model Training
Loads preprocessed data, builds the LSTM model, trains it, and saves the final model and checkpoints.

Run the training script:

```powershell
python -m scripts.training
```

### Model Evaluation & Validation
Evaluates the trained model using RMSE and MAE metrics and visualizes predictions.

Run the evaluation script:

```powershell
python -m scripts.evaluation
```

### Running Unit Tests

```powershell
python -m unittest discover -s tests
```

## Common Windows 11 PowerShell Commands

- **List Files and Directories:** `ls`
- **Change Directory:** `cd <directory_name>`
- **Activate Virtual Environment:** `.\env\Scripts\Activate.ps1`
- **Run a Python Script:** `python <script_name.py>`
- **Set Execution Policy (if required):** `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`

## Configuration
Modify `config.py` to adjust:

- Data file paths
- Training split and validation split ratios
- Outlier thresholds
- Model hyperparameters (LSTM units, dropout rate, bidirectional flag, learning rate, epochs, batch size)

## Notes
- Place raw CSV data in `rawData/`
- Preprocessed data and models are saved in `data/`
- Generated evaluation plots are stored in `plots/`
- For customization or debugging, refer to scripts in `scripts/`

## License
Mukhola James, 2025

---

This README provides an overview of the project, environment setup, and command execution using Windows 11 PowerShell.