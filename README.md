
# 🧠 Diabetes Risk Predictor

This is a full-stack machine learning application that predicts whether a person is likely to have diabetes based on medical input parameters. It consists of:

- ✅ A FastAPI **backend** for predictions  
- ✅ A Streamlit **frontend** for user interaction  
- ✅ SHAP-based **explanations** for transparency  
- ✅ Deployed on **Render** (both frontend and backend)

---

## 🚀 Live Demo

- **Frontend (Streamlit):** [Visit App](https://diabetes-streamlit-frontend.onrender.com/)
- **Backend (FastAPI):** Render-deployed API endpoint (used internally by frontend)

---

## 🗂️ Project Structure

```
diabetes-risk-predictor/
├── api/                   # FastAPI backend (main.py)
├── data/                  # Raw and cleaned dataset
├── frontend/              # Streamlit frontend (streamlit_app.py)
├── models/                # Trained ML model and scaler
├── notebooks/             # Jupyter Notebooks for EDA and model building
├── src/                   # Utility scripts (e.g. model_saver.py)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation (this file)
```

---

## 🧪 Features

- Predict diabetes using Random Forest Classifier
- Interactive Streamlit UI with form-based input
- Scaled preprocessing using `StandardScaler`
- Feature importance visualization using SHAP
- Live deployment on Render (no local setup needed)

---

## ⚙️ How to Run Locally

### 1. Clone the repository:

```bash
git clone https://github.com/Ibadatt-2k/diabetes-risk-predictor.git
cd diabetes-risk-predictor
```

### 2. Install dependencies

Create a virtual environment (optional) and install packages:

```bash
pip install -r requirements.txt
```

### 3. Run the backend (FastAPI)

```bash
cd api
uvicorn main:app --reload
```

### 4. Run the frontend (Streamlit)

Open a new terminal:

```bash
cd frontend
streamlit run streamlit_app.py
```

---

## 🛠️ Deployment

### Platform: [Render](https://render.com)

- Backend: Deployed as a **Web Service** (FastAPI)
- Frontend: Deployed as a **Static Site** with `streamlit` as entry command

Ensure your frontend `streamlit_app.py` uses the deployed backend URL like:

```python
response = requests.post("https://<your-backend>.onrender.com/predict", json=input_data)
```

---

## 📊 Input Features

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  

---

## 📈 Model and Explainability

- **Model**: Random Forest Classifier  
- **Explainability**: SHAP TreeExplainer for global + local interpretation

---

## 🤝 Credits

- Dataset: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- Built by: *Ibadatt Singh Aulakh*
