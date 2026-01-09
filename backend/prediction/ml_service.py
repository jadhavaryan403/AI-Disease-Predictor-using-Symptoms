import numpy as np
import joblib
from tokenizers import Tokenizer
import onnxruntime as ort
from django.conf import settings


# --------------------------------------------------
# Disease Predictor (BERT + ANN → ONNX ONLY)
# --------------------------------------------------
class DiseasePredictor:
    def __init__(self):
        self.tokenizer = None
        self.bert_session = None
        self.ann_session = None
        self.label_encoder = None

        self._load_models()

    # --------------------------------------------------
    def _load_models(self):
        try:
            # LabelEncoder
            self.label_encoder = joblib.load(settings.LABEL_ENCODER_PATH)
            print("✅ LabelEncoder loaded")

            # Tokenizer (Rust, lightweight)
            self.tokenizer = Tokenizer.from_pretrained(
                "nlpie/tiny-clinicalbert"
            )
            print("✅ Tokenizer loaded")

            # Tiny ClinicalBERT ONNX
            self.bert_session = ort.InferenceSession(
                str(settings.BERT_ONNX_PATH),
                providers=["CPUExecutionProvider"]
            )
            print("✅ Tiny ClinicalBERT ONNX loaded")

            # ANN ONNX
            self.ann_session = ort.InferenceSession(
                str(settings.ANN_ONNX_PATH),
                providers=["CPUExecutionProvider"]
            )
            print("✅ ANN ONNX loaded")

        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    # --------------------------------------------------
    def get_bert_embedding(self, text: str) -> np.ndarray:
        encoded = self.tokenizer.encode(text)

        input_ids = encoded.ids[:512]
        attention_mask = encoded.attention_mask[:512]

        pad_len = 512 - len(input_ids)

        if pad_len > 0:
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len

        input_ids = np.array([input_ids], dtype=np.int64)
        attention_mask = np.array([attention_mask], dtype=np.int64)

        outputs = self.bert_session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )

        token_embeddings = outputs[0]  # (1, 512, hidden_dim)
        mask = attention_mask[..., None]

        summed = np.sum(token_embeddings * mask, axis=1)
        counts = np.sum(mask, axis=1)

        return (summed / counts).astype(np.float32)


    # --------------------------------------------------
    def predict_disease(self, text, top_k=5):
        embedding = self.get_bert_embedding(text)

        logits = self.ann_session.run(
            None,
            {"input": embedding}
        )[0][0]

        probs = self.softmax(logits)

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
    @staticmethod
    def softmax(x):
        exp = np.exp(x - np.max(x))
        return exp / exp.sum()

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
