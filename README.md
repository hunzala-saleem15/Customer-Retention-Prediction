# Customer Churn Prediction

This project predicts customer churn for a telecommunications company using machine learning models. It includes a Jupyter notebook for exploratory analysis & modeling, trained models, and a Flask-based web interface for interactive predictions and dashboards.

---

## 📂 Project Structure

```
Customer Churn Prediction/
├── Colab_NoteBook_Video.mp4              # Walkthrough video of the notebook
├── Customer_Churn__Prediction.ipynb      # Jupyter notebook for analysis & model training
├── Telco_customer_churn.xlsx             # Original dataset
├── Interface Implementation/             # Flask web app
│   ├── app.py                            # Main Flask app
│   ├── preprocess.py                     # Data preprocessing scripts
│   ├── utils.py                          # Helper functions
│   ├── requirements.txt                  # Python dependencies
│   ├── Run.txt                           # Instructions to run the app
│   ├── data/churn_data.csv               # Processed dataset for web app
│   ├── models/                           # Pre-trained models (.pkl)
│   ├── templates/                        # HTML templates for Flask app
│   ├── static/                           # CSS, images, and plots
│   └── ScreenShots/                      # Screenshots of the interface
└── Plots/                                # Model evaluation and EDA plots
```

---

## 🚀 Features

* **Data Analysis**: In-depth churn exploration with visualizations.
* **Machine Learning**: Logistic Regression, Random Forest, Neural Network, and Stacking Ensemble.
* **Web Interface**: User-friendly dashboard to upload data and get churn predictions.
* **Pre-trained Models**: Ready-to-use `.pkl` models for quick deployment.

---

## 🛠️ Installation

1. Clone the repository or extract the ZIP file.
2. Navigate to the `Interface Implementation` folder:

```bash
cd "Customer Churn Prediction/Interface Implementation"
```

3. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Jupyter Notebook

Open `Customer_Churn__Prediction.ipynb` to view the EDA, model building, and evaluation steps.

### Web App

Run the Flask app:

```bash
python app.py
```

Then open your browser at `http://127.0.0.1:5000` to access the dashboard.

* Upload customer data (CSV) for batch predictions.
* Compare model performance using built-in charts.
* View churn insights via interactive dashboards.

Screenshots of the interface are available in the `ScreenShots/` folder.

---

## 📊 Models Included

* Logistic Regression
* Random Forest
* Neural Network
* Stacking Ensemble

Pre-trained models are stored in `Interface Implementation/models/`.

---

## 📁 Dataset

The original Telco customer churn dataset is included as `Telco_customer_churn.xlsx`.

---

## 📝 Requirements

All dependencies are listed in `Interface Implementation/requirements.txt`.
Install them via `pip install -r requirements.txt`.

---

## 📄 License

This project is for educational purposes. Please check dataset licensing before commercial use.

---
