# 🛒 SuperKart Sales Predictor

**Capstone Project — Postgraduate Diploma in Machine Learning & Artificial Intelligence for Business Applications**  
**University of Texas at Austin, in partnership with Great Learning**

---

## 📘 Overview

**SuperKart Sales Predictor** is an end-to-end **machine learning application** that predicts product sales across multiple retail stores based on product, pricing, and store characteristics.  

This project demonstrates the full lifecycle of an applied ML solution — from **data collection and preprocessing**, to **model training, deployment, and front-end integration** using **Flask** and **Streamlit** in containerized Hugging Face Spaces.  

It was developed as the **capstone project** for the *Postgraduate Diploma in Machine Learning and Artificial Intelligence for Business Applications* at **UT Austin and Great Learning**, showcasing practical skills in supervised learning, model optimization, and full-stack deployment.

---

## 🧩 Project Architecture

<pre> SuperKart-Sales-Predictor/
|
├── backend/ # Flask API (model inference)
│ ├── app.py # Main backend app
│ ├── sales_prediction_model_v1_0.joblib
│ ├── requirements.txt
│ └── Dockerfile
│
├── frontend/ # Streamlit user interface
│ ├── app.py # Streamlit front-end
│ ├── requirements.txt
│ └── Dockerfile
│
├── SuperKart_Sales_Predictor.html # Jupyter Notebook (EDA, model training)
└── README.md # Project documentation </pre>

---

## 🚀 Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Data Collection** | Python, Pandas, NumPy, API Requests, Web Scraping |
| **Exploration & Visualization** | Matplotlib, Seaborn, Plotly |
| **Modeling** | Scikit-learn, Random Forest, XGBoost |
| **Serialization** | Joblib |
| **Backend** | Flask, REST API |
| **Frontend** | Streamlit |
| **Deployment** | Hugging Face Spaces (Docker containers) |

---

## 🧠 Data Science Workflow

### 1. **Data Preparation**
- Collected and cleaned structured retail sales data.  
- Engineered new features such as:
  - `Scaled_Product_Allocation` — normalized store space allocation.  
  - `Power_Item` — binary flag for high-revenue SKUs.  
- Encoded categorical variables (`Product_Type`, `Store_Location_City_Type`) using ordinal and one-hot encoding schemes.

### 2. **Exploratory Data Analysis**
- Conducted univariate and bivariate analyses to uncover relationships between **sales**, **price**, **store size**, and **city tier**.  
- Identified skewed features and outliers impacting model performance.  
- Key insights:
  - Price and location were the strongest predictors of sales.  
  - Product weight correlated with both pricing and sales.  
  - Data on store allocation area required cleanup for better accuracy.  

### 3. **Model Development**
- Implemented and tuned **Random Forest** and **XGBoost** regressors.  
- Optimized using **GridSearchCV** to minimize **R2 Score**.  
- Evaluated using **R2 Score**, **RMSE**, and **MAE** metrics.  
- Achieved strong predictive performance on test data (~0.95 R2 Score).  

### 4. **Model Deployment**
- Serialized the best model (`sales_prediction_model_v1_0.joblib`) for use in a **Flask API**.  
- Exposed an endpoint `/v1/predict` for real-time predictions.  
- Integrated the API with a **Streamlit** front-end interface hosted in a separate Hugging Face Space.  

---

## 🌐 Live Demo

| Component | Hugging Face Space |
|------------|-------------------|
| **Frontend (Streamlit UI)** | [SuperKart Frontend](https://huggingface.co/spaces/SpaceMonkey25/SuperKart-Sales-Predictor-Frontend) |
| **Backend (Flask API)** | [SuperKart Backend](https://huggingface.co/spaces/SpaceMonkey25/SuperKart-Sales-Predictor-Backend) |

**Backend Endpoint:**  
`https://SpaceMonkey25-SuperKart-Sales-Predictor-Backend.hf.space/v1/predict`

---

## 🎯 Business Insights

- **Price & location** are the strongest predictors of future sales.  
- **Product weight** influences consumer perception and pricing.  
- **Power Items** (top-performing SKUs) significantly drive store revenue.  
- **Store allocation space** inconsistencies suggest a need for better inventory planning.  
- **Tier 1 stores** show potential for higher price elasticity and optimization.  

---

## 📊 Results Summary - Final 6 models

| Model                       |   RMSE   |   MAE   | R-squared | Adj. R-squared |  MAPE  |
|------------------------------|:--------:|:-------:|:---------:|:--------------:|:------:|
| RF Tuned Scaled              | 329.73   | 245.72  | 0.9045    | 0.9040         | 0.0916 |
| XGB Final w/ Product_Type    | 237.96   | 97.80   | 0.9503    | 0.9501         | 0.0380 |
| XGB No Allocation            | 232.99   | 94.05   | 0.9523    | 0.9521         | 0.0382 |
| XGB Tuned No Allocation      | 226.82   | 97.69   | 0.9548    | 0.9546         | 0.0393 |
| RF Scaled                    | 222.19   | 78.99   | 0.9566    | 0.9564         | 0.0324 |
| **RF Final w/ Product_Type** |**221.17**|**78.49**| **0.9570**| **0.9568**     | **0.0324** |


The tuned **base Random Forest model** trained on scaled allocation and product types demonstrated superior generalization, capturing complex nonlinear interactions between features and achieving higher predictive accuracy on unseen data.

---

## 🧰 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/SpaceMonkey25/SuperKart-Sales-Predictor.git
cd SuperKart-Sales-Predictor

# Backend setup
cd backend
pip install -r requirements.txt
python app.py

# Frontend setup
cd ../frontend
pip install -r requirements.txt
streamlit run app.py~ 
```  

---

## 🏆 Acknowledgments

This project was developed as part of the **Postgraduate Diploma in Machine Learning & Artificial Intelligence for Business Applications** from
**The University of Texas at Austin**, in collaboration with **Great Learning**.

Special thanks to the mentors and reviewers who provided feedback on the end-to-end lifecycle — from data wrangling to deployment.

## 📫 Contact

**Adam Cunningham**
Machine Learning Engineer (in training) | Marketing & AI Strategist
[📍 LinkedIn](www.linkedin.com/in/adam-cunningham-b685553a)
[💼 Hugging Face Profile](https://huggingface.co/SpaceMonkey25)

✅ *If you found this project helpful, please consider giving it a ⭐ on GitHub!*
