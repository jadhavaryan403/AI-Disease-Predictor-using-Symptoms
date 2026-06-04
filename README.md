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
Clinical BERT Embeddings <br>
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
- Postgresql

### 🔹 Machine Learning & NLP
- PyTorch 
- Hugging Face Transformers
- Scikit-learn
- NumPy

---
### Home Page
<img width="1900" height="1015" alt="Home page" src="https://github.com/user-attachments/assets/ba5487ae-1e63-42a0-808f-2b1fdd19779b" />

### Dashboard Page
<img width="1911" height="950" alt="Dashboard page" src="https://github.com/user-attachments/assets/04bdee02-22e2-4c18-9feb-3b4f76f11e7a" />

### Patient Profile
<img width="1870" height="1012" alt="Patient profile page" src="https://github.com/user-attachments/assets/41734f70-8f74-4913-92b6-ac8e2057cdef" />

### Upload Medical Note
<img width="1907" height="930" alt="Medical note upload page" src="https://github.com/user-attachments/assets/20bba339-0dff-40c9-af78-817c70ac4a0c" />

### Disease Prediction Result
<img width="1889" height="1019" alt="Prediction page" src="https://github.com/user-attachments/assets/d202d03d-6a12-4ae7-8906-965bc3fef4cc" />

---

### Check the Website live on this link
https://ai-disease-predictor-using-symptoms.onrender.com

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/medical-disease-predictor.git
cd medical-disease-predictor
