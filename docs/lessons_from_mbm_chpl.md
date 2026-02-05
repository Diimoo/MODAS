# Lessons from MBM & CHPL Applied to MODAS

## Overview

MODAS incorporates key lessons learned from two previous projects:
- **MBM** (Memory-Based Model): Reinforcement learning with Hebbian plasticity
- **CHPL** (Compositional Hebbian Prototype Learning): Cross-modal binding attempts

---

## Lessons from MBM

### What Worked ✓

#### 1. Meta-plasticity
```python
# Reduces learning rate for overused units
effective_lr = base_lr / (1 + beta * usage_count)
```
**Benefit**: Prevents collapse where few units dominate.

**Applied in MODAS**:
- V1 dictionary learning
- A1 dictionary learning
- ATL prototype updates

#### 2. Usage Tracking
```python
# Track how much each unit is used
usage_count += activation.abs()
```
**Benefit**: Monitors capacity utilization.

**Applied in MODAS**:
- All modules track usage
- Used for meta-plasticity
- Reported in validation

#### 3. Unit-norm Constraints
```python
# Keep prototypes/bases normalized
prototypes = F.normalize(prototypes, dim=1)
```
**Benefit**: Prevents magnitude explosions, stable learning.

**Applied in MODAS**:
- V1 dictionary bases
- A1 dictionary bases
- ATL prototypes
- Language embeddings

#### 4. Capacity Monitoring
```python
effective_capacity = (usage_count > threshold).sum() / total
if capacity < 0.5:
    print("WARNING: Low capacity!")
```
**Benefit**: Early warning for collapse.

**Applied in MODAS**:
- Logged during training
- Part of validation metrics

### What Didn't Work ✗

#### 1. Trace Decay
```python
# DON'T DO THIS
trace = decay * trace + (1 - decay) * activation
```
**Problem**: Killed capacity, created instability.

**Not used in MODAS**.

#### 2. Weight Meta-plasticity
```python
# DON'T DO THIS
lr_weight = base_lr / (1 + beta * weight_magnitude)
```
**Problem**: Hurt performance.

**Not used in MODAS** - only usage-based meta-plasticity.

#### 3. Normalized Correlation
```python
# DON'T DO THIS
sim = (x @ y) / (x.norm() * y.norm() + eps)
```
**Problem**: Worse than standard cosine similarity.

**MODAS uses**: Standard cosine similarity.

---

## Lessons from CHPL

### What Worked ✓

#### 1. Temporal Memory for Hard Negatives
```python
# Buffer stores recent samples
memory_buffer = deque(maxlen=100)

# Sample negatives from buffer
for neg in random.sample(memory_buffer, k=20):
    neg_sim = similarity(query, neg)
    neg_sims.append(neg_sim)
```
**Benefit**: Better contrastive signal.

**Applied in MODAS**:
- ATL maintains separate memory per modality
- Samples 20 negatives per update

#### 2. Contrastive Margin
```python
# Not just similarity, but margin over negatives
margin = pos_sim - max(neg_sims)
```
**Benefit**: Actual discrimination signal.

**Applied in MODAS**:
- ATL computes margin
- Used as learning modulator

#### 3. Three-factor Hebbian
```python
# pre × post × modulator
update = pre_activation * post_activation * modulator
```
**Benefit**: Directionality in learning.

**Applied in MODAS**:
- ATL uses three-factor rule
- Modulator = tanh(margin)

#### 4. Diagnostic Tests
```python
# Always compare to random baseline
learned_disc = compute_discrimination(learned_model)
random_disc = compute_discrimination(random_model)
improvement = learned_disc - random_disc
```
**Benefit**: Catches artifacts early.

**Applied in MODAS**:
- ATL validation includes random baseline test
- Warns if improvement < 0.05

### What Didn't Work ✗

#### 1. Softmax over Prototypes
```python
# DON'T DO THIS
activations = softmax(similarities / temperature)
```
**Problem**: Creates ~0.7 artifact, loses discrimination.

**MODAS uses**: Sigmoid activation instead.
```python
# DO THIS
activations = sigmoid(similarities / temperature)
```

#### 2. Winner-Takes-All
```python
# DON'T DO THIS
activations = one_hot(argmax(similarities))
```
**Problem**: Too sparse, no generalization.

**MODAS uses**: Soft activations with temperature.

#### 3. Pure Hebbian Without Modulation
```python
# DON'T DO THIS
delta = pre * post  # No modulator!
```
**Problem**: No direction, strengthens everything.

**MODAS uses**: Always modulated Hebbian.

#### 4. Predictive Coding Alone
```python
# DON'T DO THIS
loss = prediction_error.pow(2).mean()
```
**Problem**: Error minimization ≠ discrimination.

**MODAS uses**: Contrastive objective with margin.

---

## Summary Table

| Mechanism | MBM | CHPL | MODAS |
|-----------|-----|------|-------|
| Meta-plasticity | ✓ | - | ✓ |
| Usage tracking | ✓ | - | ✓ |
| Unit-norm | ✓ | ✓ | ✓ |
| Temporal memory | - | ✓ | ✓ |
| Contrastive margin | - | ✓ | ✓ |
| Three-factor Hebbian | - | ✓ | ✓ |
| Sigmoid (not softmax) | - | ✓ | ✓ |
| Random baseline test | - | ✓ | ✓ |
| Trace decay | ✗ | - | ✗ |
| Winner-takes-all | - | ✗ | ✗ |

---

## Implementation References

### Meta-plasticity in V1
```python
# modas/modules/v1_sparse_coding.py
effective_lr = self.eta / (1.0 + self.meta_beta * self.usage_count)
self.dictionary += (effective_lr.unsqueeze(1) * delta)
```

### Temporal Memory in ATL
```python
# modas/modules/atl_semantic_hub.py
self.memory_vis: deque = deque(maxlen=memory_size)
self.memory_lang: deque = deque(maxlen=memory_size)
```

### Sigmoid Activation in ATL
```python
# modas/modules/atl_semantic_hub.py
activations = torch.sigmoid(similarities / self.temperature)
# NOT: activations = F.softmax(similarities / self.temperature, dim=-1)
```

### Three-factor Hebbian in ATL
```python
# modas/modules/atl_semantic_hub.py
modulator = torch.tanh(margin)
delta = effective_lr * modulator * (hebbian_vis + hebbian_lang) / 2
```

---

## Key Takeaways

1. **Meta-plasticity is essential** for preventing capacity collapse
2. **Never use softmax** for prototype activation (use sigmoid)
3. **Always use contrastive margin**, not just similarity
4. **Three-factor Hebbian** provides learning direction
5. **Compare to random baseline** to catch artifacts
6. **Monitor capacity** throughout training
