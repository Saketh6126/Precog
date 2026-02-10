# PreCog: Understanding and Mitigating Spurious Correlations in CNNs

A comprehensive investigation into spurious correlation learning in convolutional neural networks, focusing on how models exploit color-digit associations in a synthetically biased MNIST dataset.

## Overview

This project systematically studies shortcut learning in deep neural networks through iterative experimentation. We develop interpretability techniques, implement debiasing strategies, and analyze adversarial robustness to understand how neural networks make decisions beyond simple accuracy metrics.

**Key Achievement**: Developed a robust model achieving **85.2% accuracy** on a deliberately inverted test set, compared to a baseline biased model that achieves only **24.3%** (worse than random chance).

## Tasks Completed

| Task | Description | Status | Key Results |
|------|-------------|--------|------------|
| **Task 0** | Biased Dataset Generation | ✅ Complete | 95% color-digit correlation |
| **Task 1** | CheaterCNN Training | ✅ Complete | 96.8% train, 24.3% test |
| **Task 2** | Neural Visualization | ✅ Complete | Polysemantic neurons identified |
| **Task 3** | Grad-CAM Implementation | ✅ Complete | From-scratch, validated |
| **Task 4a** | Gradient Penalty | ✅ Complete | 73% test accuracy (+48.7%) |
| **Task 4b** | Counterfactual Training | ✅ Complete | **85.2% test accuracy (+60.9%)** |
| **Task 5** | Adversarial Robustness | ✅ Complete | 1.5× more robust |

## Project Structure

```
Precog/
├── TECHNICAL_REPORT.tex          # Full technical documentation
├── README.md                      # This file
├── 
├── Task0&1.ipynb                 # Dataset generation & biased model
├── Task2.ipynb                   # Neural visualization
├── Task3.ipynb                   # Grad-CAM implementation
├── Task4a.ipynb                  # Gradient penalty debiasing
├── Task4b.ipynb                  # Counterfactual training
├── Task4b_dataset.ipynb          # Counterfactual dataset generation
├── Task5.ipynb                   # Adversarial robustness
│
├── Models/
│   └── cheater_cnn3_24_fg.pth    # Biased model weights
│
├── Robust_Models/
│   └── cnn3_24v2_85              # Robust model (85.2% hard test)
│
├── Data/
│   ├── Raw/                      # Original MNIST
│   ├── Processed_Fg/             # Foreground stroke colored
│   └── Processed_Fg_Counterfactuals/  # Paired dataset
│
├── channel_visualizations/       # Activation maximization outputs
│   ├── layer_0/, layer_2/, layer_4/
│
└── Class/                        # Pre-split class data
    └── mnist_class_0.npy ... mnist_class_9.npy
```

## Key Findings

### 1. Shortcut Learning is Architecture-Dependent
- High-capacity models (32-64-128 channels) learn both color and shape → 75-85% hard test accuracy
- Low-capacity CheaterCNN (8-16-16 channels) exploits only color → **24.3% hard test accuracy**
- **Insight**: Capacity constraints force feature selection toward shortcuts

### 2. Foreground Color ≫ Background Color
- Background coloring: 60-70% hard test accuracy (weak shortcut)
- Foreground stroke coloring: 24.3% hard test accuracy (strong shortcut)
- **Insight**: Shortcuts must align with model inductive biases to be exploited

### 3. Color Dominates Throughout Network Hierarchy
- **Layer 0**: Global color gradients with minimal spatial structure
- **Layer 2**: Color-texture mixtures, no digit outlines emerge
- **Layer 4**: Polysemantic neurons encoding multiple unrelated features
- **Insight**: Shortcut learning operates at the representation level, not just decision boundary

### 4. Explicit Constraints Outperform Regularization
- Gradient penalty: 73% accuracy (implicit regularization)
- Counterfactual pairs: **85.2% accuracy** (explicit invariance)
- **Insight**: When spurious correlations are extremely strong (95%), models need direct evidence that shortcuts are incorrect, not merely penalties for using them

### 5. Debiasing Improves Adversarial Robustness
- Biased model: 99% attack success at ε=12.75 (only 10 PGD steps)
- Robust model: 68% attack success at ε=12.75 (even with 30 PGD steps)
- **Result**: Models learning causal features develop more robust decision boundaries

### 6. Shortcut Robustness ≠ Domain Generalization
- Both biased and robust models achieve ~10% accuracy on vanilla MNIST (3-channel converted)
- **Insight**: Our interventions target color correlations within Colored-MNIST distribution, not general domain invariance

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Precog

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision scikit-learn numpy matplotlib jupyter

# Optional: For Grad-CAM visualization
pip install grad-cam
```

### Running Notebooks

```bash
jupyter notebook

# Open and run notebooks in order:
# 1. Task0&1.ipynb - Generate dataset and train biased model
# 2. Task2.ipynb - Visualize learned representations
# 3. Task3.ipynb - Implement Grad-CAM
# 4. Task4a.ipynb - Gradient penalty debiasing
# 5. Task4b.ipynb - Counterfactual training
# 6. Task5.ipynb - Adversarial robustness
```

### Loading Pre-trained Models

```python
import torch
from model_definition import CheaterCNN  # Define your model class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load biased model
biased_model = CheaterCNN().to(device)
biased_model.load_state_dict(torch.load('Models/cheater_cnn3_24_fg.pth', map_location=device))
biased_model.eval()

# Load robust model
robust_model = CheaterCNN().to(device)
robust_model.load_state_dict(torch.load('Robust_Models/cnn3_24v2_85', map_location=device))
robust_model.eval()
```

## Methodology

### Dataset Design

**Training Set (Easy):**
- 60,000 colored MNIST images
- 95% bias: Digit color matches class
- 5% counter-examples: Random wrong color
- 3-channel RGB, 28×28 resolution

**Test Set (Hard):**
- 10,000 images
- 0% bias: Color inverted (correlation broken)
- Tests whether model learned shape or just color

### CheaterCNN Architecture

| Layer | Type | Kernel | Stride | Output |
|-------|------|--------|--------|--------|
| Conv1 | Conv2d | 9×9 | 2 | 8×14×14 |
| Conv2 | Conv2d | 5×5 | 2 | 16×7×7 |
| Conv3 | Conv2d | 3×3 | 1 | 16×7×7 |
| FC | Linear | - | - | 10 classes |

**Total Parameters:** ~8,000

**Design Philosophy:**
- Large kernels (9×9, 5×5) aggregate color information
- Aggressive striding discourages fine-grained shape learning
- Low capacity creates information bottleneck

### Debiasing Methods

#### Task 4a: Gradient Penalty
```
Loss = CE(f(x), y) + λ · ||∇_x f_y(x)||²
```
- Penalizes input sensitivity
- Indirect signal for color-invariance
- **Result**: 73% hard test accuracy

#### Task 4b: Counterfactual Consistency
```
Loss = CE(f(x), y) + λ_cf · ||f(x) - f(x_cf)||²
```
- Paired training with color-inverted examples
- Direct supervision: same shape, different color → same label
- **Result**: 85.2% hard test accuracy
- **Hyperparameter**: λ_cf = 0.5, 2 warmup epochs, 10 total epochs
- **Key Insight**: Method is relatively robust to moderate changes in λ_cf

## Interpretability & Visualization

### 1. Activation Maximization
Optimize input images to maximize channel activations in specific layers:
- Initialize with noise
- Gradient ascent on channel activation
- L2 regularization + Gaussian blur for interpretability

**Finding**: All layers show color dominance with minimal shape structure

### 2. Grad-CAM (From Scratch)
Mathematical formulation:
```
α_k^c = (1/HW) Σ ∂y^c/∂A_ij^k
Heatmap = ReLU(Σ α_k^c · A^k)
```
**Implementation**:
- Forward hooks capture activation maps
- Backward hooks capture gradients
- Global average pooling → channel weights
- Weighted sum + ReLU + normalization

**Validation**: Closely aligns with pytorch-gradcam library

**Finding**: Biased models show diffuse attention on colored regions; robust models focus on digit strokes

### 3. Polysemanticity Analysis
Individual neurons respond to multiple unrelated features:
- Example: Layer 2, Channel 9 activates for:
  - Cyan color (digit 5)
  - Vertical line structures (digits 1, 7)
- **Cause**: Information bottleneck (16 channels encoding 100+ semantic features)
- **Theory**: Aligns with Superposition Hypothesis from mechanistic interpretability

## Adversarial Robustness

### PGD Attack Formulation
```
x^(0) = x + Uniform(-ε, ε)
x^(i+1) = Π_ε(x^(i) + α · sign(∇_x L(f(x^(i)), t)))
```
**Parameters**:
- ε = 12.75 (L∞, ~5% of pixel range)
- α = 1.5 (step size)
- Steps: 10 (biased model), 30 (robust model)

### Results

| Model | Attack Success | Target Confidence | ε Required |
|-------|---------------|-------------------|-----------|
| Biased | 99% | 99% | ~5 (way below budget) |
| Robust | 68% | 67% | >20 (exceeds budget) |

**Key Insight**: Debiasing and adversarial robustness are complementary objectives—both benefit from learning invariant, semantically meaningful features.

## Critical Insights

### The Nature of Shortcut Learning
Biased model achieves 24.3% on hard test set—**worse than random chance (10%)**. The model learns P(ŷ=c | color=c) ≈ 0.95 on training data, but hard test inverts this correlation. This demonstrates a critical failure mode in real-world deployments where models performing well on biased validation sets can catastrophically fail under distribution shift.

### Explicit > Implicit for Strong Biases
When spurious correlations are extremely strong (95%):
- Regularization methods provide indirect, weak signals
- Counterfactual pairs offer explicit evidence
- **Principle**: Models need proof that shortcuts are incorrect, not merely punishment for using them

### Polysemanticity Under Capacity Constraints
With only 16 channels but 100+ semantic features to encode:
- Neurons multiplex representations
- No clean disentanglement of features
- Suggests architectures need sufficient capacity OR explicit invariance constraints

## Limitations & Future Work

**Current Limitations**:
- Single architecture tested (CheaterCNN, 8-16-16 channels)
- Low-resolution dataset (28×28 MNIST)
- Limited hyperparameter exploration
- Primarily empirical observations

**Future Directions**:
- Scale to complex datasets (CIFAR-10, CelebA, medical imaging)
- Formalize counterfactual training theory with PAC-learning guarantees
- Sparse autoencoders for polysemanticity decomposition
- Certified robustness analysis (randomized smoothing)
- Advanced debiasing: Group DRO, EIIL, Concept Bottleneck Models

## References

1. **Geirhos et al. (2020)** - Shortcut Learning in Deep Neural Networks (Nature MI)
2. **Selvaraju et al. (2017)** - Grad-CAM: Visual Explanations from Deep Networks (ICCV)
3. **Arjovsky et al. (2019)** - Invariant Risk Minimization
4. **Sagawa et al. (2020)** - Distributionally Robust Neural Networks (ICLR)
5. **Madry et al. (2018)** - Towards Deep Learning Models Resistant to Adversarial Attacks (ICLR)
6. **Anthropic (2022)** - Toy Models of Superposition (Mechanistic Interpretability)
