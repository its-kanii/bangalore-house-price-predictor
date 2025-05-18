# 🏠 Bangalore House Price Predictor

This is a machine learning-powered web application built using Streamlit that predicts real estate prices in Bengaluru. It enables users to input property details such as location, square footage, number of bedrooms and bathrooms, and receive an estimated price based on a trained regression model.

🔍 Demo
View the live app locally:
[http://localhost:8501](http://localhost:8501)

📷 Screenshot


---

## 📌 Features

* 💡 Predict house prices based on location, size, and amenities
* 🗂 Cleaned and preprocessed housing dataset with location encoding
* 📊 Trained machine learning model (Random Forest)
* 🧮 Handles missing values and rare locations
* 🖥️ User-friendly interface built using Streamlit

---

## 🧠 Model & Dataset

* Dataset: Bangalore Housing dataset (Kaggle or CSV file)
* Preprocessing:

  * Filled missing values (bath, balcony)
  * Extracted numerical square footage
  * One-hot encoded locations
  * Removed outliers using z-score
* Model: Random Forest Regressor (scikit-learn)
* Evaluation: R² score and cross-validation

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/its-kanii/bangalore-house-price-predictor.git
cd bangalore-house-price-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```


⚙️ Project Structure
```
bangalore-house-price-predictor/
├── data/
│   └── Bengaluru_House_Data.csv
├── model.py
├── app.py
├── house_price_model.pkl
├── columns.json
├── requirements.txt
└── README.md
```

---
## 📈 Example Prediction

Input:

* Location: Electronic City
* Size: 1200 sqft
* Bedrooms: 2
* Bathrooms: 2

Output:

* Estimated Price: ₹54.8 Lakhs

---

## 💡 Possible Enhancements

* Integrate AI for predictive analytics & feature importance
* Use NLP for textual feature processing
* Enable geolocation or map-based input
* Deploy online via Streamlit Cloud or Hugging Face Spaces

---

## 🙋‍♀️ Author

Developed by @its-kanii
🔗 LinkedIn: [https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/kanimozhi-kathirvel)
📧 Email: [Kanimozhi](mailto:kanimeena678@gmail.com)

---

## 📄 License

This project is licensed under the MIT License – see the LICENSE file for details.


