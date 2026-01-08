# 🏥 AI-Powered Medical Disease Prediction System

An end-to-end **AI-based medical disease prediction web application** designed for doctors to manage patients and predict from around 20 diseases from unstructured medical notes — all wrapped inside a **Django web interface**.

---

## 🚀 Project Overview

Doctors often write unstructured medical notes that are difficult to analyze programmatically.  
This project solves that problem by:

The system allows doctors to:
- 👨‍⚕️ Register and manage patient records
- 🗂️ Store patient medical notes securely
- 📄 Upload medical notes for each patient
- 🤖 Predict possible diseases using transformer-based NLP models
- 📊 View model confidence scores and top disease predictions

By combining **patient data management** with **AI-driven disease prediction**, the application acts as a unified platform for both **clinical record handling** and **intelligent analysis**.

---

## 🧠 Key Features

### 🧑‍⚕️ Doctor & Patient Management
✔ Doctor-oriented workflow  
✔ Add, edit, and manage patient records  
✔ Store patient medical notes and history  
✔ Secure handling of patient-related data 

### 🤖 AI Disease Prediction
✔ Transformer-based clinical text embeddings  
✔ Disease prediction using ML / ANN  
✔ Top-N disease predictions with confidence scores  
✔ Ready for deployment (Docker / Cloud / Hugging Face)

---

## Evaluation metrics

| Metric         | Value           |
| -------------- | --------------- |
| Accuracy       | ~92%            |
| Recall         | ~93%            |
| Top-3 Accuracy | ~99%            |

<img width="649" height="547" alt="Confusion matrix" src="https://github.com/user-attachments/assets/5d75fd61-12d3-4802-b025-1362542c25bf" />

---

## 🏗️ System Architecture

Medical Note From doctor <br>
        ↓<br>
Text Preprocessing <br>
        ↓<br>
Clinical BERT Embeddings
        ↓<br>
ANN Classifier<br>
        ↓<br>
Disease Prediction<br>
        ↓<br>
Django Web Interface

---

## 🧰 Tech Stack

### 🔹 Backend & Web
- **Django**
- Django Templates
- Django ORM

### 🔹 Machine Learning & NLP
- PyTorch 
- Hugging Face Transformers
- Scikit-learn
- NumPy

---
### Home Page
<img width="1849" height="1013" alt="{Home Page}" src="https://github.com/user-attachments/assets/da0e143d-ffee-4a86-b196-f37de75eed67" />

### Dashboard Page
<img width="1862" height="1006" alt="Dashboard page" src="https://github.com/user-attachments/assets/eb7c2722-4742-4995-a60b-a5538a2df396" />

### Patient Profile
<img width="1893" height="1006" alt="Patient profile" src="https://github.com/user-attachments/assets/8358bb0e-2592-4437-bdab-a1f66ebfe767" />

### Upload Medical Note
<img width="1915" height="987" alt="Medical image upload page" src="https://github.com/user-attachments/assets/98d537bd-06ef-464a-98e1-0884cf2b09f2" />

### Disease Prediction Result
<img width="1827" height="1010" alt="Final Prediction" src="https://github.com/user-attachments/assets/11b768ce-2311-43f2-8648-b833e2b4b14d" />

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
In bash
git clone https://github.com/your-username/medical-disease-predictor.git
cd medical-disease-predictor
