# Module 2 — Visual / Image Forensics

## 1. Overview & Purpose
Module 2 is the **Visual / Image Forensics** component of the AI-Based Fake Identity & Document Screening System. It receives original, full-resolution document images from previous modules (which have already completed hard-pass validation, OCR, QR-code validation, and MRZ validation).

Module 2 operates at the **case level** and evaluates:
1. **Module 2.1: AI-Generated Image Detection Gate**: A sequential case-level security gate. If **any** document in the case is detected as AI-generated, Module 2 stops immediately, raises a `RED_FLAG`, leaves subsequent documents unprocessed, aborts Module 2.2 forensics, and returns `forensic_pass_score = null`.
2. **Module 2.2: Visual / Image Forensics**: When all declared documents pass the AI-generation gate, Module 2 performs two forensic analyses:
   - **Splice / Tamper Forensics**: Detection of digital manipulation, splicing, and forgery using an EfficientNet-B3 binary classifier trained on the SIDTD dataset.
   - **Guilloché / Background Pattern Forensics**: Comparative verification of background security patterns against indexed authentic references using Siamese-style embedding similarity.

---

## 2. Architecture

```text
                                  CASE INPUT
               (uuid, documents_present, original document images)
                                      │
                                      ▼
                             Input Validation
                         (Compulsory vs Optional)
                                      │
                                      ▼
                        Read `documents_present`
                                      │
                                      ▼
                        MODULE 2.1: SEQUENTIAL AI GATE
                   Deterministic order: passport → visa → national_id
                             (→ driving_license → permit)
                                      │
             ┌────────────────────────┴────────────────────────┐
             ▼                                                 ▼
     AI_GENERATED = YES                                AI_GENERATED = NO
             │                                                 │
             ▼                                                 ▼
          RED FLAG                                    Check next document
      STOP ENTIRE CASE                                         │
    No Module 2.2 executed                            Repeat for all docs
 forensic_pass_score = null                                    │
             │                                                 ▼
             ▼                                    MODULE 2.2: FORENSICS
        Risk Engine                       ┌────────────────────┴────────────────────┐
                                          ▼                                         ▼
                                  Splice / Tamper                         Guilloché / Pattern
                                   (EfficientNet-B3)                      (Reference Comparison)
                                          │                                         │
                                          └────────────────────┬────────────────────┘
                                                               │
                                                               ▼
                                                      Forensic Pass Scorer
                                                        (Default: 60/40)
                                                               │
                                                               ▼
                                                          Risk Engine
```

---

## 3. Input Contract
Module 2 receives case-level inputs via `CaseInput` or dictionary format:

| Field | Type | Requirement | Description |
| :--- | :--- | :--- | :--- |
| `uuid` | string | **Compulsory** | Unique case identifier |
| `documents_present` | list of strings | **Compulsory** | Declared list of available documents |
| `passport_image` | Image / Path / bytes | **Compulsory** | Original passport image |
| `visa_image` | Image / Path / bytes | **Compulsory** | Original visa image |
| `national_id_image` | Image / Path / bytes | **Compulsory** | Original national ID image |
| `driving_license_image` | Image / Path / bytes | *Optional* | Optional driving license image |
| `permit_image` | Image / Path / bytes | *Optional* | Optional permit image |

### Valid Case Configurations
- **3 documents**: `["passport", "visa", "national_id"]`
- **4 documents**: `["passport", "visa", "national_id", "driving_license"]` OR `["passport", "visa", "national_id", "permit"]`
- **5 documents**: `["passport", "visa", "national_id", "driving_license", "permit"]`

Documents are processed strictly in this deterministic sequence:
1. `passport`
2. `visa`
3. `national_id`
4. `driving_license` (if present)
5. `permit` (if present)

Absent optional documents are neither waited for nor processed.

---

## 4. Output Contract

### A. RED FLAG (Early Stop on AI-Generated Image)
When any document is identified as AI-generated:
```json
{
  "uuid": "case-uuid-12345",
  "status": "RED_FLAG",
  "ai_generation_check": "FAILED",
  "flag_type": "AI_GENERATED_DOCUMENT",
  "flagged_document": "national_id",
  "forensic_pass_score": null
}
```

### B. COMPLETED (Full Forensics Evaluated)
When all declared documents pass the AI gate:
```json
{
  "uuid": "case-uuid-12345",
  "status": "COMPLETED",
  "ai_generation_check": "PASSED",
  "forensic_pass_score": 100.0,
  "documents_analyzed": [
    "passport",
    "visa",
    "national_id"
  ],
  "document_results": {
    "passport": {
      "document_type": "passport",
      "ai_generated": false,
      "ai_gate_status": "PASSED",
      "tamper_result": {
        "status": "BONA_FIDE",
        "label": "BONA_FIDE",
        "passed": true,
        "confidence": 0.94,
        "forgery_probability": 0.06,
        "bona_fide_probability": 0.94
      },
      "guilloche_result": {
        "status": "CONSISTENT",
        "passed": true,
        "similarity_score": 0.88,
        "reference_available": true
      },
      "forensic_score": 100.0
    }
  }
}
```

---

## 5. Module 2.1 — AI-Generated Image Detection Gate
- Acts as a sequential **case gate**.
- Uses an independent preprocessing pipeline (`original -> resize -> tensor -> normalize`) that never mutates or replaces original document images.
- Returns internal binary decision: `AI_GENERATED = YES` or `NO`.
- Does **not** expose synthetic probabilities to the caller.
- Implemented as an abstract modular interface (`BaseAIDetector`, `ModularAIDetector`), ready to drop in dedicated deep learning detectors without changing the orchestrator.

---

## 6. Module 2.2 — Visual / Image Forensics

### A. Splice & Tamper Forensics
- **Architecture**: EfficientNet-B3 binary classifier.
- **Model Target**: Document forgery / manipulation classification (`BONA_FIDE` vs `FORGED`).
- **Forensic Preprocessing**:
  - **Error Level Analysis (ELA)**: Computes pixel differential against controlled recompressed JPEG representations at configurable quality (default 90).
  - **Discrete Cosine Transform (DCT)**: Block-based 8x8 2D frequency analysis extracting DC energy, AC energy, high-frequency ratios, and AC coefficient variance.

### B. Guilloché & Background Pattern Forensics
- **Purpose**: Verifies fine-line security patterns against authentic reference templates.
- **Reference Manager**: Indexes authentic references by `(country, document_type, version, region)`.
- **Status Codes**:
  - `CONSISTENT`: Reference exists and cosine similarity $\ge$ threshold.
  - `INCONSISTENT`: Reference exists and cosine similarity < threshold.
  - `REFERENCE_REQUIRED`: No authentic reference template exists in the library. (Never converted into `INCONSISTENT` or marked fake).
  - `NOT_APPLICABLE`: Document type does not possess guilloché security patterns.

---

## 7. Forensic Scoring Logic
- The forensic pass score is **NOT** an authenticity probability.
- It is a score out of 100 representing the **percentage of currently defined weighted forensic checks that passed**.
- Default weights:
  - **Tamper Forensics**: 60%
  - **Guilloché Forensics**: 40%
- Single document scenarios:
  - `Tamper PASS` + `Guilloche PASS` = **100/100**
  - `Tamper PASS` + `Guilloche FAIL` = **60/100**
  - `Tamper FAIL` + `Guilloche PASS` = **40/100**
  - `Tamper FAIL` + `Guilloche FAIL` = **0/100**
- Missing reference handling (`REFERENCE_REQUIRED`):
  - Under default policy (`REWEIGHT`), the score renormalizes across available checks.
- If AI gate detects any AI-generated document: `forensic_pass_score = null`.

---

## 8. Important Scientific Distinction: SIDTD vs AI Generation

> [!IMPORTANT]
> - **SIDTD is a document forgery dataset** containing `BONA_FIDE` and `FORGED` identity documents.
> - **SIDTD is NOT an AI-generation dataset.**
> - The EfficientNet-B3 model trained on SIDTD is strictly used for **Module 2.2 Splice/Tamper Detection**.
> - Module 2.1 (AI-generation gate) is separated as an independent interface and is NOT trained on SIDTD.

---

## 9. Installation & Dependencies

```bash
# Clone or navigate to module2
cd module2

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `torch` >= 2.0.0
- `torchvision` >= 0.15.0
- `numpy` >= 1.23.0
- `opencv-python` >= 4.7.0
- `Pillow` >= 9.5.0
- `PyYAML` >= 6.0
- `scikit-learn` >= 1.2.0

---

## 10. Configuration
All parameters are managed in `config/module2_config.yaml`:
- Model checkpoint paths
- Classification and similarity thresholds
- Forensic weights (60/40)
- ELA quality and amplification scale
- Reference library root
- Hardware device settings (`auto`, `cuda`, `mps`, `cpu`)

---

## 11. Running Tests

Execute the complete test suite:
```bash
PYTHONPATH="src" python3 -m unittest discover -s tests -p "test_*.py" -v
```

Test coverage includes:
- `test_input.py`: 3, 4, 5-document cases, missing compulsory documents, invalid types, absent optionals.
- `test_tamper.py`: Model loading, inference interface, structured output, invalid input handling.
- `test_guilloche.py`: Reference found, missing reference (`REFERENCE_REQUIRED`), not applicable, consistent vs inconsistent.
- `test_scoring.py`: All 100/60/40/0 scoring combinations, policy edge cases, case aggregation.
- `test_module2.py`: End-to-end integration tests for Cases 1 through 6, verifying early stop and red flags.

---

## 12. Running Inference

### Via Python API:
```python
from module2.main import run_module2

case_input = {
    "uuid": "case-example-001",
    "documents_present": ["passport", "visa", "national_id"],
    "passport_image": "/path/to/passport.jpg",
    "visa_image": "/path/to/visa.jpg",
    "national_id_image": "/path/to/id.jpg",
    "metadata": {
        "country": "USA"
    }
}

result = run_module2(case_input)
print(result)
```

### Via CLI:
```bash
PYTHONPATH="src" python3 -m module2.main --input /path/to/case.json --output /path/to/results.json
```

---

## 13. Training the SIDTD Tamper Model

Training code is strictly segregated in `training/` and does not execute during inference:
```bash
python3 training/train_sidtd.py --config training/training_config.yaml
```

Features:
- Uses predefined SIDTD partition: `split_normal`.
- Binary classification: `BONA_FIDE` vs `FORGED`.
- Automatic device selection (CUDA GPU / Apple MPS / CPU fallback).
- Mixed precision support (`torch.cuda.amp`).
- Checkpointing, early stopping, and AdamW + CosineAnnealingLR.

---

## 14. Limitations & Risk Engine Integration
- **Guilloché Reference Library**: When reference patterns are not yet registered for a country/document combination, Module 2 returns `REFERENCE_REQUIRED`. It does not invent authenticity verdicts.
- **AI-Generation Gate**: Currently operates with standard image frequency/heuristic checks and an extensible interface; replace with a dedicated production model once trained on verified synthetic document datasets.
- **Risk Engine Decision**: Module 2 outputs raw forensic pass scores (0–100) and explicit flag statuses. It does not determine final business rules (Allow, Review, Deny), which remain the sole responsibility of the downstream Risk Engine.
