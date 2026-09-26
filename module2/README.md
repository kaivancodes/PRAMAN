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
- **Checkpoint Metadata Saved in `sidtd_efficientnet_b3.pth`**:
  - `loss_function`: `{'name': 'CrossEntropyLoss', 'loss_type': 'CrossEntropyLoss', 'reduction': 'mean', 'details': 'CrossEntropyLoss()'}`
  - `confusion_matrix`: Validation confusion matrix `matrix` (`[[TN, FP], [FN, TP]]`), counts (`true_negatives`, `false_positives`, `false_negatives`, `true_positives`), and metrics (`precision`, `recall`, `f1_score`, `roc_auc`).
  - `metrics`: Validation loss, validation accuracy, F1 score, ROC AUC.
  - `test_metrics`: Held-out test split evaluation loss, accuracy, and test confusion matrix.
  - `state_dict`: Full model weights.
  - `epoch`, `model_name`, `num_classes`, `class_names`, `config`.

---

## 14. Limitations & Risk Engine Integration
- **Guilloché Reference Library**: When reference patterns are not yet registered for a country/document combination, Module 2 returns `REFERENCE_REQUIRED`. It does not invent authenticity verdicts.
- **AI-Generation Gate**: Currently operates with standard image frequency/heuristic checks and an extensible interface; replace with a dedicated production model once trained on verified synthetic document datasets.
- **Risk Engine Decision**: Module 2 outputs raw forensic pass scores (0–100) and explicit flag statuses. It does not determine final business rules (Allow, Review, Deny), which remain the sole responsibility of the downstream Risk Engine.

## Model Training Architecture

### Tamper / Splice Model
EfficientNet-B3 is trained as a binary classifier using five input channels: RGB + ELA + DCT. CASIA-v2.0 and SIDTD provide forged supervision; MIDV-2020 contributes bona-fide document imagery without inventing forged labels. Raw document preprocessing is applied before forensic representation generation. RGB, ELA, and DCT receive the same spatial augmentation so their pixels remain aligned.

### Guilloché Model
Siamese ResNet-18 is trained on authentic/security-pattern imagery from MIDV-2020 + SIDTD. It learns an embedding space using CosineEmbeddingLoss and compares document pattern crops against authentic references using cosine similarity.

### Evaluation
training/evaluate_models.py is the ML evaluation pipeline. It uses held-out data and reports accuracy, precision, recall, F1, ROC-AUC, and confusion matrix for the tamper model, plus similarity-based metrics for Guilloché. The tests/ directory separately tests software correctness and training/representation regressions.

### Production Model Artifacts
Only these trained artifacts are expected for Module 2.2:
- models/efficientnet_b3_combined_tamper.pth
- models/guilloche_siamese_resnet18.pth


## GPU Training Only

The GPU machine does not need the production runtime to train Module 2.

Copy/clone only:
- `module2/training/`
- `module2/src/module2/models/`
- `module2/src/module2/preprocessing/ela.py`
- `module2/src/module2/preprocessing/dct.py`
- `module2/src/module2/guilloche/pattern_extractor.py`
- `module2/requirements.txt`
- the datasets under `module2/data/`

### Tamper training

```bash
cd PRAMAN/module2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python training/train_tamper.py \
  --config training/tamper_training_config.yaml \
  --device cuda
```

Output:
```
models/efficientnet_b3_combined_tamper.pth
models/tamper_training_history.json
```

### Guilloché training

```bash
python training/train_guilloche.py \
  --config training/guilloche_training_config.yaml \
  --device cuda
```

Output:
```
models/guilloche_siamese_resnet18.pth
models/guilloche_training_history.json
```

### Dry-run before full GPU training

```bash
python training/train_tamper.py --config training/tamper_training_config.yaml --device cuda --dry-run
python training/train_guilloche.py --config training/guilloche_training_config.yaml --device cuda --dry-run
```

### What is saved in the Tamper checkpoint

- EfficientNet-B3 weights
- 5-channel input declaration: RGB + ELA + DCT
- binary class names
- manipulation-reason class names
- Tamper/reason thresholds
- best validation metrics
- training configuration

### What is saved in the Guilloché checkpoint

- Siamese ResNet-18 weights
- 128-D embedding dimension
- pattern crop size
- model input size
- ImageNet normalization declaration
- similarity threshold
- best validation metrics
- training configuration

Production input validation is intentionally NOT part of these training scripts. It belongs to the common Module 2 upload gate before the queue item enters the two forensic branches.

### SIDTD dataset on the GPU

The provided Google Drive file can be downloaded on the GPU using its Drive file ID:

```bash
cd PRAMAN/module2
mkdir -p data
pip install gdown
gdown "https://drive.google.com/uc?id=1b3zxOM_7eEhs5h3MxNdrRFmfVWT1ym33" -O data/SIDTD_download
```

If the downloaded file is an archive, extract it under `module2/data/` so that the training config can resolve:

```
data/
├── split_normal/
│   ├── train_split_SIDTD.csv
│   ├── val_split_SIDTD.csv
│   └── test_split_SIDTD.csv
├── templates/
│   └── Images/
│       ├── reals/
│       └── fakes/
└── ...
```

Do not commit the dataset to GitHub. Keep it on the GPU filesystem and keep `data/` ignored.


## Production Forensic Pipeline

For each document that reaches Module 2, the production path is:

```text
Real uploaded image
        ↓
Common input validation / upstream gate
        ↓
Document boundary detection
        ↓
Document cropping
        ↓
Perspective correction
        ↓
CLAHE / contrast normalization
        ↓
        ┌──────────────────────────────┐
        │                              │
        ↓                              ↓
   TAMPER BRANCH                  GUILLOCHÉ BRANCH
        │                              │
   RGB + ELA + DCT             Security-pattern extraction
        │                              │
   EfficientNet-B3              Crop / normalize pattern
        │                              │
   Tamper Result                 Siamese ResNet-18
        │                              │
        └──────────────┬───────────────┘
                       ↓
              Combine Results
                       ↓
                Forensic Score
                       ↓
                  Risk Engine
```

The two forensic branches run in parallel after common document normalization. A technical error remains an `ERROR`/incomplete result and is not converted into a forensic `FAIL`.

### Training/evaluation consistency

Training and held-out evaluation use the same core deterministic document normalization before branch-specific preprocessing. The production upload-quality gate itself is not part of dataset training.

Tamper: `dataset image → common document normalization → RGB + ELA + DCT → EfficientNet-B3`

Guilloché: `dataset image → common document normalization → security-pattern extraction → crop/normalize → Siamese ResNet-18`
