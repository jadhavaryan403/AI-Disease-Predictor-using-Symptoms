import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 🔒 Force CPU

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel
from django.conf import settings


# --------------------------------------------------
# ANN MODEL (must match training architecture)
# --------------------------------------------------
class ANNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.01, eps=0.001)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64, momentum=0.01, eps=0.001)

        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32, momentum=0.01, eps=0.001)

        self.fc6 = nn.Linear(32, num_classes)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.bn3(x)

        x = F.relu(self.fc4(x))
        x = self.bn4(x)

        x = F.relu(self.fc5(x))
        x = self.bn5(x)

        x = self.fc6(x)   # ❗ NO softmax here
        return x


# --------------------------------------------------
# Disease Predictor
# --------------------------------------------------
class DiseasePredictor:
    def __init__(self):
        self.device = torch.device("cpu")

        self.tokenizer = None
        self.bert_model = None
        self.ann_model = None
        self.label_encoder = None

        self.load_models()

    # --------------------------------------------------

    def load_models(self):
        """Load PyTorch ANN, LabelEncoder, Tiny ClinicalBERT"""
        try:
            # LabelEncoder
            self.label_encoder = joblib.load(settings.LABEL_ENCODER_PATH)
            num_classes = len(self.label_encoder.classes_)
            print("✅ LabelEncoder loaded")

            # Tiny ClinicalBERT (PyTorch)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "nlpie/tiny-clinicalbert"
            )
            self.bert_model = AutoModel.from_pretrained(
                "nlpie/tiny-clinicalbert"
            ).to(self.device)
            self.bert_model.eval()
            print("✅ Tiny ClinicalBERT loaded (PyTorch, CPU)")

            # ANN Classifier
            self.ann_model = ANNClassifier(
                input_dim=312,   # Tiny ClinicalBERT hidden size
                num_classes=num_classes
            ).to(self.device)

            self.ann_model.load_state_dict(
                torch.load(settings.MODEL_PATH, map_location=self.device)
            )
            self.ann_model.eval()
            print("✅ ANN model loaded (PyTorch)")

        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    # --------------------------------------------------

    @torch.no_grad()
    def get_bert_embedding(self, text):
        """
        Mean pooled Tiny ClinicalBERT embedding
        Output shape: (1, 312)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        summed = torch.sum(token_embeddings * attention_mask, dim=1)
        counts = torch.sum(attention_mask, dim=1)

        embedding = summed / counts
        return embedding

    # --------------------------------------------------

    @torch.no_grad()
    def predict_disease(self, text, top_k=5):
        embedding = self.get_bert_embedding(text)

        logits = self.ann_model(embedding)
        preds = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_indices = np.argsort(preds)[-top_k:][::-1]
        diseases = self.label_encoder.inverse_transform(top_indices)

        results = []
        for idx, disease in zip(top_indices, diseases):
            results.append({
                "disease": disease,
                "confidence": round(float(preds[idx]) * 100, 2)
            })

        return {
            "top_predictions": results,
            "best_prediction": results[0]["disease"],
            "best_confidence": results[0]["confidence"]
        }

    # --------------------------------------------------

    def process_medical_note(self, note_text=None):
        if not note_text or len(note_text.strip()) < 20:
            raise ValueError("Note is too short for accurate results")

        return self.predict_disease(note_text.strip())


# --------------------------------------------------
# Singleton (Django safe)
# --------------------------------------------------
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = DiseasePredictor()
    return _predictor
