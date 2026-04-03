# Agent 2: Diagnosis Agent

**Role:** Clinical Reasoning — RAG over Medical Knowledge  
**Type:** A2A Agent  
**Framework:** LangChain + LangGraph + FastAPI  
**Protocol:** Receives PatientState from Orchestrator via A2A

---

## What This Agent Does

The Diagnosis Agent performs **differential diagnosis** — the clinical process of narrowing down possible conditions from a patient's symptoms, history, and presentation. It does this by combining:

1. Structured analysis of the patient's FHIR data (existing conditions, medications, demographics)
2. RAG (Retrieval-Augmented Generation) over a medical knowledge base to ground its reasoning in clinical guidelines

It returns a **ranked list of probable diagnoses** with confidence scores and clinical justification, formatted as FHIR `Condition` resources. Its output is also passed to the Consensus Agent to detect disagreements with the Lab Analysis Agent.

---

## Responsibilities

1. Parse `PatientState` to extract symptoms, current conditions, and medical history
2. Formulate a clinical query from the patient's presentation
3. Retrieve relevant clinical guidelines, ICD-10 descriptions, and diagnostic criteria from the vector database
4. Run LLM reasoning over retrieved context + patient data
5. Return a ranked differential diagnosis list with confidence scores
6. Output structured FHIR `Condition` resources for each probable diagnosis

---

## Input

Receives `PatientState` object from the Orchestrator (output of Agent 1).

```json
{
  "patient_state": { "...full PatientState object..." },
  "chief_complaint": "Fever, productive cough, shortness of breath for 3 days"
}
```

---

## Output

```json
{
  "differential_diagnosis": [
    {
      "rank": 1,
      "icd10_code": "J18.9",
      "display": "Community-acquired pneumonia",
      "confidence": 0.87,
      "supporting_evidence": ["Productive cough", "Fever 38.9°C", "High WBC", "Age 54"],
      "against_evidence": ["No consolidation confirmed on imaging yet"],
      "fhir_condition": { "resourceType": "Condition", "code": {...}, "...": "..." }
    },
    {
      "rank": 2,
      "icd10_code": "J22",
      "display": "Acute lower respiratory tract infection",
      "confidence": 0.61,
      "supporting_evidence": ["Cough", "Fever"],
      "against_evidence": ["WBC elevation suggests bacterial, not viral"]
    }
  ],
  "top_diagnosis": "Community-acquired pneumonia",
  "confidence_level": "HIGH",
  "reasoning_summary": "Patient presents with classic triad of community-acquired pneumonia: fever, productive cough, and elevated WBC. Age and lack of recent hospitalization suggest community-acquired rather than hospital-acquired pathogen.",
  "recommended_next_steps": ["Chest X-ray confirmation", "Sputum culture", "Blood cultures if sepsis suspected"]
}
```

---

## How It Works — Step by Step

### Step 1: Build Clinical Query
Extract the most clinically relevant features from `PatientState`:
```python
query = f"""
Patient: {age}y {gender}
Chief complaint: {chief_complaint}
Active conditions: {[c['display'] for c in active_conditions]}
Current medications: {[m['drug'] for m in medications]}
Allergies: {[a['substance'] for a in allergies]}
Abnormal labs: {[l for l in labs if l['flag'] in ['HIGH','LOW','CRITICAL']]}
"""
```

### Step 2: RAG Retrieval
Query ChromaDB for the top-k most relevant clinical documents:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
docs = retriever.get_relevant_documents(query)
```

**Knowledge base contents to index:**
- ICD-10-CM clinical descriptions (free, CDC)
- NIH clinical practice guidelines (free, PubMed)
- Merck Manual diagnostic criteria summaries
- Symptom-to-diagnosis mappings from medical textbooks

### Step 3: LLM Reasoning Chain
Use LangChain with a structured output parser:
```python
chain = (
    {"context": retriever, "patient_data": RunnablePassthrough()}
    | diagnosis_prompt
    | llm
    | JsonOutputParser()
)
result = chain.invoke(query)
```

### Step 4: Confidence Scoring
Apply rule-based adjustment on top of LLM output:
- Boost confidence if lab values confirm the diagnosis
- Reduce confidence if patient has allergies to first-line treatment (clinical red flag)
- Flag "LOW" confidence if differential list is very wide (>5 near-equal options)

### Step 5: Format as FHIR Conditions
```python
fhir_condition = {
    "resourceType": "Condition",
    "subject": {"reference": f"Patient/{patient_id}"},
    "code": {
        "coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": icd10_code, "display": display}]
    },
    "verificationStatus": {"coding": [{"code": "provisional"}]},
    "clinicalStatus": {"coding": [{"code": "active"}]}
}
```

---

## Knowledge Base Setup

### Indexing Pipeline
```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

loader = DirectoryLoader("./medical_docs/", glob="**/*.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split_documents(docs)
vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db")
```

### Sources to Use (All Free)
| Source | Content | Format |
|---|---|---|
| CDC ICD-10-CM | Diagnostic codes + descriptions | CSV/TXT |
| NIH MedlinePlus | Disease descriptions, symptoms | Web scrape |
| OpenMRS clinical concepts | Clinical terminology | JSON |
| Synthea clinical notes | Synthetic patient narratives | Text |

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | GPT-4o-mini or Claude Haiku (cost-efficient) |
| RAG framework | LangChain |
| Vector database | ChromaDB (local) or Pinecone |
| Embedding model | `text-embedding-3-small` (OpenAI) |
| Output parsing | LangChain JsonOutputParser |
| API framework | FastAPI |

---

## Prompt Design

```
You are a clinical decision support system performing differential diagnosis.

RETRIEVED CLINICAL GUIDELINES:
{context}

PATIENT DATA:
{patient_data}

Analyze this patient and return a JSON differential diagnosis list.
For each diagnosis include:
- ICD-10 code and display name
- Confidence score (0.0 to 1.0)
- Supporting evidence from patient data
- Evidence against this diagnosis
- Brief clinical reasoning

Return ONLY valid JSON. No preamble.
```

---

## Your Existing Skills That Apply

- RAG pipeline architecture (from WellAI and Gemma RAG System projects)
- LangChain document loaders and retrievers
- ChromaDB vector database
- Structured output parsing with LangChain
- Prompt engineering for structured JSON output

---

## Disagreement Signal

This agent outputs `confidence` scores. The Consensus Agent (Agent 7) compares this agent's top diagnosis against the Lab Analysis Agent's conclusion. If the top diagnosis codes differ AND combined confidence delta > 0.3, a disagreement is flagged and escalated.