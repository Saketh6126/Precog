# PreCog: Understanding and Mitigating Spurious Correlations in CNNs

A comprehensive research assignment investigating how convolutional neural networks learn spurious correlations, and techniques to mitigate them.

## Overview

This project systematically studies shortcut learning in deep neural networks using a synthetically biased MNIST dataset. Through iterative experimentation, we develop interpretability techniques, debiasing strategies, and adversarial robustness analysis to understand how neural networks make decisions beyond simple accuracy metrics.

**Key Achievement**: Developed a robust model achieving **85-96% accuracy** on a deliberately inverted test set, compared to a baseline biased model that achieves only **24.3%**.

## Tasks Completed

| Task | Description | Status | Key Results |
|------|-------------|--------|------------|
| **Task 0** | Biased Dataset Generation | ✅ Complete | 95% color-digit correlation |
| **Task 1** | CheaterCNN Training | ✅ Complete | 96.8% train, 24.3% test |
| **Task 2** | Neural Visualization (Activation Maximization) | ✅ Complete | Polysemantic neurons identified |
| **Task 3** | Grad-CAM Implementation (from scratch) | ✅ Complete | Validated against pytorch-gradcam |
| **Task 4a** | Gradient Penalty Debiasing | ✅ Attempted | 73% test accuracy (+48.7%) |
| **Task 4b** | Counterfactual Consistency Training | ✅ Complete | 85.2% test accuracy (+60.9%) |
| **Task 5** | Adversarial Robustness Analysis | ✅ Complete | 1.5× more robust |

## Project Structure

```
Precog/
├── TECHNICAL_REPORT.tex          # Full technical documentation (LaTeX)
├── README.md                      # This file
├── 
├── Notebooks/
│   ├── Task0&1.ipynb             # Dataset generation & biased model training
│   ├── Task2.ipynb               # Neural network visualization
│   ├── Task3.ipynb               # Grad-CAM implementation
│   ├── Task4a.ipynb              # Gradient penalty debiasing
│   ├── Task4b.ipynb              # Counterfactual consistency training
│   ├── Task4b_dataset.ipynb      # Counterfactual dataset generation
│   └── Task5.ipynb               # Adversarial robustness analysis
│
├── Models/
│   ├── cheater_cnn3_24_fg        # Biased model weights
│   ├── cheater_cnn3_24_fg.pth    # PyTorch format
│   └── ...                        # Other trained models
│
├── Robust_Models/
│   ├── cnn3_24_v2_5_2_10_85_96   # Best robust model (85.2% hard test)
│   └── ...
│
├── Data/
│   ├── Raw/                       # Original MNIST
│   ├── Processed_Fg/             # Foreground stroke colored
│   ├── Processed_Fg_Counterfactuals/  # Paired dataset for Task 4b
│   └── ...                        # Other processed variants
│
├── channel_visualizations/        # Activation maximization outputs
│   ├── layer_0/                  # Early layer channels
│   ├── layer_2/                  # Intermediate layer channels
│   └── layer_4/                  # Deep layer channels
│
├── Visualizations/
│   ├── Pallete.png               # Color palette mapping
│   ├── Background_Coloring.jpg   # Failed approach
│   ├── Foreground_coloring.png   # Successful approach
│   └── Confusion_Matrix_Biased.png  # Misclassification patterns
│
└── class/                         # Pre-computed class data
```

## Key Findings

### 1. **Shortcut Learning is Architecture-Dependent**
- High-capacity models (32-64-128 channels) learn both color and shape → 68.4% hard test accuracy
- Low-capacity CheaterCNN (8-16-16 channels) exploits only color → 24.3% hard test accuracy
- **Insight**: Capacity constraints force feature selection toward shortcuts

### 2. **Foreground Color ≫ Background Color**
- Background coloring: 60-70% hard test (weak shortcut)
- Foreground stroke coloring: 24.3% hard test (strong shortcut)
- **Insight**: Shortcuts must align with model inductive biases to be exploited

### 3. **Color Dominates Network Representations**
- Layer 0: Global color gradients dominate
- Layer 2: Color-texture mixtures, no digit structure
- Layer 4: Polysemantic neurons encoding multiple unrelated features
- **Insight**: Shortcut learning operates at the representation level, not just decision boundary

### 4. **Counterfactual Training Outperforms Gradient Penalty**
- Gradient penalty: 73% (regularization-based)
- Counterfactual pairs: 85.2% (explicit invariance)
- **Insight**: While gradient penalty shows significant improvement, explicit constraints achieve superior robustness

### 5. **Adversarial Robustness Correlates with Debiasing**
- Biased model: Needs ε=10.5 for successful attack (below budget)
- Robust model: Needs ε=18.3 for successful attack (exceeds budget)
- **Result**: 1.5× more robust at task-specified ε=12.75

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

# Load biased model
biased_model = torch.load('Models/cheater_cnn3_24_fg.pth')

# Load robust model
robust_model = torch.load('Robust_Models/cnn3_24_v2_5_2_10_85_96.pth')
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
- **Best hyperparameter**: λ_cf = 0.5, 2 warmup epochs, 10 total epochs

## Interpretability Techniques

### 1. Activation Maximization
- Optimize input to maximize channel activations
- Visualize what each layer's neurons respond to
- L2 regularization + periodic Gaussian blur for interpretability

**Finding**: All layers dominated by color, minimal shape structure

### 2. Grad-CAM (Gradient-weighted Class Activation Mapping)
```
α_k^c = (1/HW) Σ ∂y^c/∂A_ij^k
Heatmap = ReLU(Σ α_k^c · A^k)
```
- Implementation: From-scratch using PyTorch hooks
- Validation: 0.98+ correlation with pytorch-gradcam library

**Finding**: Biased model attends to distributed color regions; robust model focuses on digit strokes

### 3. Polysemanticity Analysis
- Layer 2, Channel 9 activates for:
  - Cyan color (digit 5)
  - Vertical line structures (digits 1, 7)
- **Cause**: Information bottleneck (16 channels vs. 100+ semantic features)
- **Theory**: Aligns with Superposition Hypothesis from mechanistic interpretability

## Adversarial Robustness

### PGD Attack Setup
- **Target**: Misclassify digit 7 as digit 3
- **Budget**: ε = 12.75 (L∞, ~5% of pixel range)
- **Steps**: 30 iterations, α = 1.5

### Results

| Model | Success | Confidence | ε Required |
|-------|---------|------------|-----------|
| Biased | ✓ YES | 96.3% | 10.5 |
| Robust | ✗ NO | 67.0% | 18.3 |

**Transferability**: Adversarial examples don't transfer between models (different features)

## Critical Insights

### 1. The Nature of Shortcut Learning
- Biased model: 24.3% on hard test (worse than random!)
- Reason: Model learns P(ŷ=c \| color=c) ≈ 0.95, but hard test inverts correlation
- **Real-world implication**: Models performing well on biased validation can catastrophically fail

### 2. Explicit > Implicit for Strong Biases
- Regularization (gradient penalty) is indirect and weak
- Counterfactual pairs provide explicit evidence
- **Principle**: When bias is 95%, models need proof it's wrong, not punishment for using it

### 3. Polysemanticity Under Capacity Constraints
- Information bottleneck creates multiplexed representations
- No disentangled semantic features
- Suggests architectures need sufficient capacity OR explicit constraints (counterfactuals)

## Technical Details

### Grad-CAM Implementation Highlights
- **Forward hook**: Capture activation maps A^k at target layer
- **Backward hook**: Capture gradients ∂y^c/∂A^k during backprop
- **Pooling**: Global average of gradients → channel importance weights
- **Upsampling**: Bilinear interpolation to input resolution

### Counterfactual Dataset Generation
```python
# For each training sample (x, y):
x_colored = apply_biased_color(x, y)      # 95% rule
x_cf = apply_random_color(x)              # Any random color
# Train to minimize: ||f(x_colored) - f(x_cf)||²
```

## Limitations & Future Work

### Limitations
- Single architecture tested (CheaterCNN)
- Low-resolution dataset (28×28 MNIST)
- Limited hyperparameter search
- Mostly empirical observations

### Future Directions
- Scale to CIFAR-10, CelebA, medical imaging
- Formalize counterfactual training theory
- Sparse autoencoders for polysemanticity decomposition
- Certified robustness guarantees
- Group DRO, EIIL, and other advanced debiasing methods

## References

Key papers this work builds on:

1. **Geirhos et al. (2020)** - Shortcut Learning in Deep Neural Networks
2. **Selvaraju et al. (2017)** - Grad-CAM: Visual Explanations
3. **Arjovsky et al. (2019)** - Invariant Risk Minimization
4. **Sagawa et al. (2020)** - Distributionally Robust Neural Networks
5. **Anthropic (2022)** - Toy Models of Superposition (mechanistic interpretability)

## Citation

If you use this work, please cite:

```bibtex
@assignment{precog2026,
  title={Biased Model Analysis: Understanding and Mitigating Spurious Correlations in CNNs},
  author={Saketh},
  year={2026},
  institution={PreCog Research Group}
}
```

## Contact & Discussion

For questions or discussion about specific implementations, please refer to the individual Jupyter notebooks which contain detailed comments and experimental logs.

---

**Last Updated**: February 9, 2026  
**Status**: Complete ✅  
**Total Effort**: 7 comprehensive tasks, 6 Jupyter notebooks, 1000+ line technical report
