import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel
from django.conf import settings


# --------------------------------------------------
# ANN MODEL (INFERENCE SAFE)
# --------------------------------------------------
class ANNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)

        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)

        self.fc6 = nn.Linear(32, num_classes)

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

        return self.fc6(x) 


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

        self._load_models()


    # --------------------------------------------------

    def _load_models(self):
        try:
            # Load LabelEncoder
            self.label_encoder = joblib.load(settings.LABEL_ENCODER_PATH)
            num_classes = len(self.label_encoder.classes_)
            print("✅ LabelEncoder loaded")

            # Load Tiny ClinicalBERT (CPU)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "nlpie/tiny-clinicalbert"
            )
            self.bert_model = AutoModel.from_pretrained(
                "nlpie/tiny-clinicalbert"
            )
            self.bert_model.to(self.device)
            self.bert_model.eval()
            print("✅ Tiny ClinicalBERT loaded (CPU)")

            # Load ANN
            self.ann_model = ANNClassifier(
                input_dim=312,
                num_classes=num_classes
            )
            self.ann_model.load_state_dict(
                torch.load(settings.MODEL_PATH, map_location="cpu")
            )
            self.ann_model.to(self.device)
            self.ann_model.eval()
            print("✅ ANN model loaded (CPU)")

        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")


    # --------------------------------------------------

    @torch.no_grad()
    def get_bert_embedding(self, text):
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

        return summed / counts


    # --------------------------------------------------

    @torch.no_grad()
    def predict_disease(self, text, top_k=5):
        embedding = self.get_bert_embedding(text)

        logits = self.ann_model(embedding)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_indices = np.argsort(probs)[-top_k:][::-1]
        diseases = self.label_encoder.inverse_transform(top_indices)

        results = [
            {
                "disease": disease,
                "confidence": round(float(probs[idx]) * 100, 2)
            }
            for idx, disease in zip(top_indices, diseases)
        ]

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
