from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Load BERT-based text classification pipeline once at startup
# This uses a pretrained model fine-tuned on political text
classifier = pipeline(
 "text-classification",
 model="bucketresearch/politicalBiasBERT", # pretrained on political left/right
 tokenizer="bucketresearch/politicalBiasBERT"
)

LABEL_MAP = {
 "LEFT": "Democrat",
 "RIGHT": "Republican",
 # Add more if the model has more classes
}

class InputText(BaseModel):
 input_texts: str

app = FastAPI()

@app.get("/health")
def get_health():
 return {"status": "OK"}

@app.post("/get-prediction/")
def get_prediction(input_data: InputText):
 result = classifier(input_data.input_texts, truncation=True, max_length=512)
 label = result[0]["label"]
 score = result[0]["score"]

 return {
 "prediction": LABEL_MAP.get(label, label),
 "confidence": round(score, 4)
 }

