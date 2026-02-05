# MODAS Checkpoint Criteria

## Overview

Each MODAS phase has clear GO/NO-GO criteria. **Do not proceed** if criteria are not met.

---

## Checkpoint 1: V1 Sparse Coding

### Criteria

| Metric | Threshold | Check |
|--------|-----------|-------|
| Gabor Score | > 0.6 | Filters resemble Gabor functions |
| Sparsity | > 85% | Most neurons inactive |
| Cross-similarity | < 0.5 | Codes discriminate inputs |
| Reconstruction | Reasonable | Can decode patches |

### Validation Code

```python
from modas.evaluation import validate_v1

passed, metrics = validate_v1(
    model_path="experiments/phase1_v1/checkpoints/v1_final.pt",
    test_data=test_patches,
    device="cuda",
    save_dir="experiments/phase1_v1/validation"
)

if passed:
    print("✓ V1 PASSED - Continue to Phase 2")
else:
    print("✗ V1 FAILED - Debug before proceeding")
```

### Decision Tree

```
Gabor > 0.6?
├── YES: Check sparsity
│   ├── Sparsity > 85%?
│   │   ├── YES: PASS → Continue to A1
│   │   └── NO: Adjust λ_sparse, retrain
│   └── 
└── NO: Check training
    ├── Enough epochs? → Train longer
    ├── Good data? → Check patch quality
    └── LCA converging? → Increase iterations
```

---

## Checkpoint 2: A1 Sparse Coding

### Criteria

| Metric | Threshold | Check |
|--------|-----------|-------|
| Temporal Score | > 0.3 | Filters show structure |
| Sparsity | > 85% | Most neurons inactive |
| Cross-similarity | < 0.5 | Codes discriminate inputs |

### Decision Tree

```
Temporal > 0.3?
├── YES: Check sparsity
│   ├── Sparsity > 85%?
│   │   ├── YES: PASS → Continue to ATL
│   │   └── NO: Adjust λ_sparse
│   └── 
└── NO: 
    ├── OPTION A: Skip audio, do V1↔Language only
    └── OPTION B: Debug A1 preprocessing
```

---

## Checkpoint 3: ATL Binding (CRITICAL)

### Criteria

| Discrimination | Status | Action |
|----------------|--------|--------|
| > 0.2 | EXCELLENT | 🎉 Publish, trimodal |
| 0.15 - 0.2 | GOOD | ✓ Publish, trimodal |
| 0.1 - 0.15 | MARGINAL | ⚠ Analyze, improve |
| < 0.1 | FAIL | ✗ Document, stop |

### Additional Checks

| Check | Expected |
|-------|----------|
| Improvement over random | > 0.05 |
| Capacity | > 50% |
| Swap generalization | > 70% (if applicable) |

### Validation Code

```python
from modas.evaluation import validate_atl

status, metrics = validate_atl(
    atl_path="experiments/phase4_atl_binding/checkpoints/atl_final.pt",
    v1_path="experiments/phase1_v1/checkpoints/v1_final.pt",
    test_data=test_pairs,
    device="cuda",
    save_dir="experiments/phase4_atl_binding/validation"
)

# status is one of: "EXCELLENT", "GOOD", "MARGINAL", "FAIL"
```

### Decision Tree

```
Discrimination > 0.1?
├── YES: 
│   ├── > 0.2? → EXCELLENT: Publish + Trimodal
│   ├── > 0.15? → GOOD: Publish + Trimodal
│   └── > 0.1? → MARGINAL: Try improvements
│       ├── Increase epochs
│       ├── Adjust temperature
│       ├── More hard negatives
│       └── If still marginal after 3 attempts → Document + Publish negative
└── NO: FAIL
    ├── Check V1 codes are diverse
    ├── Check language embeddings make sense
    ├── Verify binding mechanism
    └── Document negative result
```

---

## Checkpoint 4: Trimodal (Optional)

### Prerequisites
- ATL checkpoint must be GOOD or EXCELLENT
- A1 checkpoint must be PASS (or skip audio)

### Criteria

| Metric | Threshold |
|--------|-----------|
| Cross-modal retrieval | > 60% |
| Audio-visual binding | > 0.1 discrimination |

---

## Failure Analysis

### If V1 Fails

1. **Check data quality**
   - Patches normalized?
   - Diverse images?
   - Correct patch size?

2. **Check hyperparameters**
   - λ_sparse too high/low?
   - LCA iterations sufficient?
   - Learning rate appropriate?

3. **Check implementation**
   - Gram matrix correct?
   - Normalization applied?
   - Meta-plasticity working?

### If A1 Fails

1. **Check preprocessing**
   - Spectrogram parameters?
   - Audio sample rate?
   - Segment length appropriate?

2. **Check audio quality**
   - Clean speech?
   - Consistent volume?

### If ATL Fails

1. **Check input features**
   - V1 codes discriminative?
   - Language embeddings sensible?
   
2. **Check binding mechanism**
   - Using sigmoid (not softmax)?
   - Memory buffer working?
   - Contrastive margin computed correctly?

3. **Check learning**
   - Prototypes updating?
   - Meta-plasticity active?
   - Capacity not collapsing?

---

## Recording Results

### Success Report Template

```markdown
# MODAS Experiment Report

## Summary
- Date: YYYY-MM-DD
- Phase: [1-5]
- Status: [PASS/FAIL/MARGINAL]

## Metrics
- V1 Gabor: X.XX
- V1 Sparsity: XX%
- A1 Temporal: X.XX
- ATL Discrimination: X.XX

## Key Findings
[Description]

## Next Steps
[Actions]
```

### Failure Report Template

```markdown
# MODAS Failure Analysis

## Summary
- Date: YYYY-MM-DD
- Phase: [1-5]
- Status: FAIL

## Metrics at Failure
[Metrics]

## Root Cause Analysis
[Analysis]

## Attempted Fixes
1. [Fix 1] - Result: [Result]
2. [Fix 2] - Result: [Result]

## Conclusions
[What we learned]

## Recommendations
[Future directions]
```
