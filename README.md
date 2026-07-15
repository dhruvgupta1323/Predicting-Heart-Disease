# 🫀 Cardiac Risk Monitor

A Streamlit web app that estimates the probability of heart disease presence from 13 clinical measurements, using a trained Logistic Regression model on the Statlog Heart Disease dataset.

---

---

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Clinical input form** — demographics, vitals, and cardiac test results, grouped into clear sections
- **Live vitals summary** — a sidebar-style panel that updates in real time as you fill in the form
- **Risk gauge** — a color-coded probability gauge (green / amber / red) instead of a plain number
- **Explainability** — shows the top clinical factors pushing an individual prediction up or down, derived from the model's own coefficients
- **Resilient loading** — if the model files aren't found next to the app, you can upload them directly from the sidebar

 

## Demo
![App demo](assets/demo.gif)

## Tech stack

- [Streamlit](https://streamlit.io/) — UI
- [scikit-learn](https://scikit-learn.org/) — model (Logistic Regression, StandardScaler)
- [Plotly](https://plotly.com/python/) — risk gauge visualization
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling

## Project structure

```
.
├── app.py                     # Streamlit application
├── style.css                  # Custom UI styling
├── requirements.txt           # Python dependencies
├── scaler.pkl                 # Fitted StandardScaler
├── heart_disease_model.pkl    # Trained LogisticRegression model
├── label_encoder.pkl          # Label encoder for target classes
└── README.md
```

## Getting started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Make sure `scaler.pkl`, `heart_disease_model.pkl`, and `label_encoder.pkl` are in the same directory as `app.py` (they're included in this repo). If they're missing, the app lets you upload them from the sidebar instead.

## Model details

| | |
|---|---|
| Algorithm | Logistic Regression |
| Features | 13 clinical measurements |
| Classes | `Absence`, `Presence` |
| Preprocessing | StandardScaler (z-score normalization) |

**Input features:** Age, Sex, Chest pain type, Resting blood pressure, Cholesterol, Fasting blood sugar > 120 mg/dl, Resting EKG results, Max heart rate, Exercise-induced angina, ST depression, Slope of peak ST segment, Number of major vessels (fluoroscopy), Thallium stress test result.

## Deployment

This app can be deployed for free on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account
3. Select this repo, branch, and `app.py` as the entry point
4. Deploy 🚀

## Disclaimer

This project is for **educational and portfolio purposes only**. It is **not** a medical device and must **not** be used for actual clinical diagnosis or treatment decisions. Always consult a qualified healthcare professional for medical advice.

## License

This project is licensed under the [MIT License](LICENSE).
