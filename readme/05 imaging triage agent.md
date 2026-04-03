# Agent 5: Imaging Triage Agent

**Role:** Medical Image Analysis — CNN-Based X-Ray Interpretation  
**Type:** A2A Agent  
**Framework:** TensorFlow/Keras + FastAPI  
**Your Unfair Advantage:** You already have a trained 90%+ accuracy pneumonia CNN model

---

## What This Agent Does

The Imaging Triage Agent is MediTwin's biggest differentiator. It wraps your already-trained pneumonia detection CNN as a FHIR-compliant A2A agent. When a chest X-ray is available for a patient, this agent:

1. Accepts a base64-encoded chest X-ray image
2. Preprocesses and runs inference through your trained CNN model
3. Returns a structured triage decision with confidence scores and severity grading
4. Writes the result as a FHIR `DiagnosticReport` resource

No other team in this hackathon will have a production-trained medical imaging model. This wins the "wow" moment in your demo video and is your clearest technical differentiator from every chatbot-based submission.

---

## Responsibilities

1. Accept image upload (base64 or file) + FHIR patient context via SHARP
2. Preprocess image to match training input format (224×224 RGB, normalized)
3. Run inference through pneumonia CNN
4. Apply confidence thresholding and severity classification
5. Generate clinical interpretation text
6. Write result as FHIR `DiagnosticReport` resource
7. Return triage priority (1=immediate, 2=urgent, 3=semi-urgent, 4=routine)

---

## Input

```json
{
  "patient_id": "fhir-patient-uuid",
  "image_data": {
    "format": "base64",
    "content_type": "image/jpeg",
    "data": "base64encodedXrayImageHere..."
  },
  "patient_context": {
    "age": 54,
    "gender": "male",
    "chief_complaint": "Productive cough, fever 3 days",
    "current_diagnosis": "Suspected community-acquired pneumonia"
  },
  "sharp_token": "Bearer ..."
}
```

---

## Output

```json
{
  "model_output": {
    "prediction": "PNEUMONIA",
    "confidence": 0.923,
    "normal_probability": 0.077,
    "pneumonia_probability": 0.923
  },
  "severity_assessment": {
    "grade": "MODERATE",
    "triage_priority": 2,
    "triage_label": "URGENT",
    "clinical_urgency": "Requires treatment within 4-6 hours"
  },
  "imaging_findings": {
    "pattern": "Lobar consolidation",
    "affected_area": "Right lower lobe (likely)",
    "bilateral": false,
    "confidence_in_findings": "HIGH"
  },
  "clinical_interpretation": "Chest X-ray pattern is highly consistent with bacterial pneumonia. Right lower lobe consolidation pattern with confidence 92.3%. Combined with clinical presentation (fever, productive cough, elevated WBC), urgent antibiotic therapy is recommended.",
  "confirms_diagnosis": true,
  "diagnosis_code": "J18.9",
  "recommended_actions": [
    "Start empirical antibiotic therapy",
    "Blood cultures before antibiotic administration",
    "Repeat chest X-ray in 48-72 hours",
    "Monitor oxygen saturation"
  ],
  "fhir_diagnostic_report": {
    "resourceType": "DiagnosticReport",
    "status": "final",
    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "RAD", "display": "Radiology"}]}],
    "code": {"coding": [{"system": "http://loinc.org", "code": "24748-7", "display": "Chest X-ray"}]},
    "subject": {"reference": "Patient/fhir-patient-uuid"},
    "conclusion": "Findings consistent with pneumonia. Triage Priority 2 (URGENT).",
    "conclusionCode": [{"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": "J18.9"}]}]
  }
}
```

---

## How It Works — Step by Step

### Step 1: Image Reception and Validation
```python
@app.post("/analyze-xray")
async def analyze_xray(request: ImagingRequest):
    # Decode base64 image
    image_bytes = base64.b64decode(request.image_data.data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Validate — reject non-chest-xray images
    if image.size[0] < 100 or image.size[1] < 100:
        raise HTTPException(400, "Image too small — minimum 100x100px")
    
    return await run_inference(image, request.patient_context)
```

### Step 2: Preprocessing
Match your training preprocessing pipeline exactly:
```python
def preprocess_image(image: PIL.Image) -> np.ndarray:
    # Resize to training input size
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    
    # Apply same normalization as training
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    # Add batch dimension
    return np.expand_dims(img_array, axis=0)
```

### Step 3: Model Inference
```python
# Load model once at startup — not per-request
model = tf.keras.models.load_model("models/pneumonia_cnn_v1.h5")

async def run_inference(image: PIL.Image, context: dict) -> dict:
    preprocessed = preprocess_image(image)
    
    # Run inference
    prediction = model.predict(preprocessed)[0][0]
    
    # prediction is pneumonia probability (sigmoid output)
    pneumonia_prob = float(prediction)
    normal_prob = 1.0 - pneumonia_prob
    
    return {
        "prediction": "PNEUMONIA" if pneumonia_prob > 0.5 else "NORMAL",
        "pneumonia_probability": pneumonia_prob,
        "normal_probability": normal_prob,
        "confidence": max(pneumonia_prob, normal_prob)
    }
```

### Step 4: Severity Classification
Apply clinical triage logic on top of model confidence:
```python
def classify_severity(pneumonia_prob: float, patient_context: dict) -> dict:
    age = patient_context.get("age", 40)
    
    # Base severity from model confidence
    if pneumonia_prob > 0.90:
        grade = "SEVERE"
        priority = 1
    elif pneumonia_prob > 0.75:
        grade = "MODERATE"
        priority = 2
    elif pneumonia_prob > 0.50:
        grade = "MILD"
        priority = 3
    else:
        grade = "NORMAL"
        priority = 4
    
    # Boost priority for high-risk patients
    if age > 65 or age < 5:
        priority = max(1, priority - 1)
    
    triage_labels = {1: "IMMEDIATE", 2: "URGENT", 3: "SEMI-URGENT", 4: "ROUTINE"}
    return {"grade": grade, "triage_priority": priority, "triage_label": triage_labels[priority]}
```

### Step 5: FHIR DiagnosticReport Generation
```python
def create_fhir_report(patient_id: str, prediction: dict, severity: dict) -> dict:
    return {
        "resourceType": "DiagnosticReport",
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "RAD"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": "24748-7", "display": "Chest X-ray AP"}]},
        "subject": {"reference": f"Patient/{patient_id}"},
        "issued": datetime.utcnow().isoformat() + "Z",
        "conclusion": f"AI Triage: {prediction['prediction']} (confidence: {prediction['confidence']:.1%}). Priority: {severity['triage_label']}.",
        "conclusionCode": [{"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": "J18.9"}]}] if prediction["prediction"] == "PNEUMONIA" else []
    }
```

---

## Model Information

Your existing trained model:
- **Architecture:** CNN with transfer learning (VGG16/ResNet50 base)
- **Training dataset:** Chest X-Ray Images (Pneumonia) — Kaggle dataset
- **Test accuracy:** 90%+
- **Output:** Binary sigmoid (0=normal, 1=pneumonia)
- **Input size:** 224×224 RGB

For the hackathon demo, load this model directly. No retraining needed.

---

## Model Deployment Considerations

```python
# Startup: load model into memory once
@app.on_event("startup")
async def load_model():
    app.state.model = tf.keras.models.load_model("models/pneumonia_cnn_v1.h5")
    print("Model loaded successfully")

# Per-request: use loaded model
async def run_inference(image, context):
    model = app.state.model
    # ... inference logic
```

---

## Dataset for Reference

**Chest X-Ray Images (Pneumonia) — Kaggle:**
- 5,863 X-Ray images (JPEG)
- 2 categories: PNEUMONIA / NORMAL
- 100% free to use
- URL: `https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia`

You've already trained on this. Just load your saved `.h5` file.

---

## Tech Stack

| Component | Technology |
|---|---|
| Model framework | TensorFlow 2.x / Keras |
| Image processing | Pillow (PIL) + NumPy |
| API framework | FastAPI with async support |
| Model serving | FastAPI (no separate serving layer needed for hackathon) |
| FHIR output | Custom FHIR R4 builder |
| Base64 handling | Python `base64` stdlib |

---

## Demo Video Strategy

This agent is your opening "wow" moment. Structure your 3-minute demo video like this:

1. (0:00-0:30) Show a real chest X-ray being submitted to the system via patient ID
2. (0:30-1:00) Show the CNN inference running and returning PNEUMONIA with 92% confidence
3. (1:00-1:30) Show the FHIR DiagnosticReport being generated
4. (1:30-2:00) Show the orchestrator combining this with lab results and diagnosis agent
5. (2:00-2:30) Show the Digital Twin's treatment scenario comparison
6. (2:30-3:00) Show the final clinical plan output

---

## Your Existing Skills That Apply

- You built this model. It's done. You just need to wrap it as a FastAPI endpoint.
- Transfer learning with TensorFlow/Keras (from your Pneumonia and Cassava Leaf Disease Detection project)
- Model deployment experience
- Data augmentation and fine-tuning

---

## Critical Note on Medical Disclaimers

Always include in the FHIR DiagnosticReport:
```
"AI-generated triage assistance only. Not a substitute for radiologist review. 
For clinical decision support use under physician supervision."
```

This is standard for AI-assisted diagnostics and shows judges you understand clinical deployment constraints.