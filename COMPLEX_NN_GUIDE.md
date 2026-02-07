# Complex-Valued Neural Networks for Video Frame Interpolation

## Overview

This project implements a **complex-valued neural network** version of PhaseNet for video frame interpolation. Unlike the original PhaseNet that separates complex steerable pyramid coefficients into amplitude and phase components, this implementation preserves the natural complex structure throughout the entire network.

## Table of Contents

1. [Why Complex-Valued Neural Networks?](#why-complex-valued-neural-networks)
2. [Architecture](#architecture)
3. [Key Components](#key-components)
4. [Advantages](#advantages)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Usage Examples](#usage-examples)
7. [Comparison with Original PhaseNet](#comparison-with-original-phasenet)

## Why Complex-Valued Neural Networks?

### Motivation

Complex numbers are fundamental in signal processing, particularly in:
- **Fourier Analysis**: Complex exponentials form the basis
- **Phase Information**: Natural representation of periodic phenomena
- **Steerable Pyramids**: Coefficients are inherently complex-valued

### Problems with Real-Valued Separation

The original PhaseNet separates complex coefficients into:
```
amplitude = |z| = sqrt(real² + imag²)
phase = ∠z = atan2(imag, real)
```

**Issues:**
1. **Phase Wrapping**: Phase discontinuities at ±π boundaries
2. **Information Loss**: Separation breaks natural coupling between amplitude and phase
3. **Inefficiency**: Requires separate processing of amplitude and phase
4. **Non-linearity**: arctangent is highly non-linear and difficult to learn

### Complex-Valued Solution

By keeping coefficients in complex form:
```
z = real + i*imag
```

We can:
1. **Preserve Structure**: Natural representation of oriented features
2. **Handle Periodicity**: Phase wrapping handled automatically
3. **Improve Efficiency**: One complex number = two real numbers but with structural constraints
4. **Better Gradients**: Complex arithmetic provides smoother gradient flow

## Architecture

### ComplexPhaseNet Structure

```
Input: Complex Steerable Pyramid Coefficients
  ├─ Level 0 (Residual): Low-frequency content
  │   └─ ComplexPhaseNetBlock → ComplexPred
  │
  ├─ Level 1-10 (Bands): Oriented features
  │   └─ ComplexPhaseNetBlock → ComplexPred
  │       ├─ ComplexConv2d (×2)
  │       ├─ ComplexBatchNorm2d
  │       └─ ComplexLeakyReLU
  │
Output: Complex Coefficients for Interpolated Frame
```

### Network Flow

```python
# Pseudo-code for forward pass
def forward(input_real, input_imag):
    for each pyramid level:
        # 1. Normalize complex input
        normalized = normalize_complex(input_real, input_imag)
        
        # 2. Concatenate with upsampled features
        concatenated = cat([normalized, prev_features_upsampled])
        
        # 3. Process through complex block
        features_r, features_i = ComplexBlock(concatenated)
        
        # 4. Predict output coefficients
        pred_r, pred_i = ComplexPred(features)
        
        # 5. Interpolate amplitude, predict phase
        output = combine(interpolated_amp, predicted_phase)
```

## Key Components

### 1. ComplexConv2d

Complex-valued 2D convolution implementing:

```
(A + Bi) ⊗ (W_r + W_i·i) = (A⊗W_r - B⊗W_i) + (A⊗W_i + B⊗W_r)i
```

**Properties:**
- Separate weight matrices for real and imaginary parts
- Maintains complex structure through convolution
- Equivalent to 2×2 real convolution with structural constraints

### 2. ComplexBatchNorm2d

Normalizes complex-valued features using covariance matrix:

```
Σ = [σ_rr  σ_ri]
    [σ_ri  σ_ii]
```

**Process:**
1. Compute mean: `μ = E[z]`
2. Center data: `z_centered = z - μ`
3. Compute covariance components: `σ_rr, σ_ii, σ_ri`
4. Whitening transformation using Cholesky decomposition
5. Affine transformation with complex scaling

**Reference:** Trabelsi et al., "Deep Complex Networks" (ICLR 2018)

### 3. Complex Activation Functions

Multiple activation options:

#### CReLU (Component-wise ReLU)
```python
CReLU(z) = ReLU(real) + i·ReLU(imag)
```

#### modReLU (Magnitude-based)
```python
modReLU(z) = ReLU(|z| + b) · (z / |z|)
```

#### zReLU (Phase-preserving)
```python
zReLU(z) = z if (real ≥ 0 AND imag ≥ 0) else 0
```

## Advantages

### 1. **Natural Phase Handling**

**Original PhaseNet:**
```python
# Phase wrapping issues at boundaries
phase = atan2(imag, real)  # Range: [-π, π]
# Discontinuity at ±π creates learning difficulties
```

**Complex PhaseNet:**
```python
# Phase naturally handled in complex form
z = real + i*imag  # Continuous representation
# No discontinuities, smoother gradients
```

### 2. **Parameter Efficiency**

For equivalent expressiveness:
- **Real-valued**: 2N parameters (separate amplitude and phase networks)
- **Complex-valued**: N complex parameters = 2N real parameters BUT with structural constraints

The constraints in complex networks provide:
- Better generalization
- Reduced overfitting
- More efficient learning

### 3. **Improved Gradient Flow**

**Complex arithmetic provides natural gradient paths:**

```python
# Gradient of complex multiplication
∂(z₁·z₂)/∂z₁ = z₂*  (complex conjugate)

# Smoother than separated amplitude/phase gradients
∂|z|/∂z = z/(2|z|)  # Undefined at |z|=0
∂∠z/∂z = -i/(z)      # Singular at origin
```

### 4. **Orientation Equivariance**

Complex steerable pyramids naturally represent oriented features. Complex networks preserve this structure:

```python
# Rotation in image space
z_rotated = z · e^(iθ)

# Naturally handled by complex operations
# Real-valued networks must learn this relationship
```

## Mathematical Foundation

### Complex Convolution

Given input `X = X_r + iX_i` and kernel `W = W_r + iW_i`:

```
Y = X ⊗ W
Y_r = X_r ⊗ W_r - X_i ⊗ W_i
Y_i = X_r ⊗ W_i + X_i ⊗ W_r
```

**Matrix Form:**
```
[Y_r]   [W_r  -W_i] [X_r]
[Y_i] = [W_i   W_r] [X_i]
```

This is a rotation-scaling transformation in the complex plane.

### Complex Batch Normalization

For input `z = [z₁, ..., z_n]ᵀ`:

1. **Compute Statistics:**
```
μ_r = E[z_r]
μ_i = E[z_i]
Σ_rr = Var[z_r]
Σ_ii = Var[z_i]
Σ_ri = Cov[z_r, z_i]
```

2. **Whitening Matrix:**
```
V = [Σ_rr  Σ_ri]^(-1/2)
    [Σ_ri  Σ_ii]
```

3. **Normalize:**
```
z_norm = V·(z - μ)
```

4. **Affine Transform:**
```
z_out = γ·z_norm + β
```

where `γ` and `β` are learnable complex parameters.

### Phase Interpolation

**Key Insight:** Phase should be interpolated on the circle, not linearly:

**Wrong (Linear):**
```python
phase_mid = (phase_start + phase_end) / 2
# Fails at ±π boundary
```

**Right (Circular):**
```python
z_start = exp(i·phase_start)
z_end = exp(i·phase_end)
z_mid = slerp(z_start, z_end, t=0.5)  # Spherical interpolation
phase_mid = angle(z_mid)
```

Complex networks naturally handle this through complex arithmetic!

## Usage Examples

### Training

```python
from net.complex_phasenet import ComplexPhaseNet, ComplexTotalLoss

# Initialize model
model = ComplexPhaseNet(feature_dim=32).cuda()

# Initialize loss
criterion = ComplexTotalLoss(v=1.0)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Get complex pyramid coefficients
        train_real, train_imag, truth_real, truth_imag = \
            get_complex_input(batch)
        
        # Forward pass
        pred_real, pred_imag = model(train_real, train_imag)
        
        # Compute loss
        loss = criterion(truth_real, truth_imag, 
                        pred_real, pred_imag,
                        truth_img, pred_img)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Inference

```python
# Load trained model
model = ComplexPhaseNet()
model.load_state_dict(torch.load('model_complex.pth'))
model.eval()

# Process input frames
with torch.no_grad():
    # Build pyramids
    pyr_start = pyramid.build(frame_start)
    pyr_end = pyramid.build(frame_end)
    
    # Extract complex coefficients
    start_real, start_imag = extract_complex(pyr_start)
    end_real, end_imag = extract_complex(pyr_end)
    
    # Prepare input
    input_real, input_imag = prepare_input(
        start_real, start_imag,
        end_real, end_imag
    )
    
    # Predict
    pred_real, pred_imag = model(input_real, input_imag)
    
    # Reconstruct frame
    pred_coeff = convert_to_pyramid(pred_real, pred_imag)
    interpolated_frame = pyramid.reconstruct(pred_coeff)
```

### Creating Custom Complex Layers

```python
from net.complex_nn import ComplexConv2d, ComplexBatchNorm2d

class CustomComplexBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ComplexConv2d(in_ch, out_ch, 3, padding=1)
        self.bn = ComplexBatchNorm2d(out_ch)
        self.conv2 = ComplexConv2d(out_ch, out_ch, 3, padding=1)
    
    def forward(self, real, imag):
        # First convolution
        real, imag = self.conv1(real, imag)
        
        # Batch normalization
        real, imag = self.bn(real, imag)
        
        # Activation (component-wise ReLU)
        real = F.relu(real)
        imag = F.relu(imag)
        
        # Second convolution
        real, imag = self.conv2(real, imag)
        
        return real, imag
```

## Comparison with Original PhaseNet

| Aspect | Original PhaseNet | Complex PhaseNet |
|--------|------------------|------------------|
| **Representation** | Amplitude + Phase (separated) | Complex (unified) |
| **Phase Handling** | Explicit wrapping needed | Automatic |
| **Parameters** | 2 separate networks | 1 complex network |
| **Gradient Flow** | Non-linear (arctangent) | Smoother (complex) |
| **Orientation Features** | Must learn from separated | Natural preservation |
| **Training Stability** | Phase discontinuities | More stable |
| **Computational Cost** | Similar | Similar |
| **Memory Usage** | Similar | Similar |

### Performance Characteristics

**Expected Improvements:**
1. **Better interpolation quality** near motion boundaries
2. **Smoother temporal consistency** in video sequences
3. **Faster convergence** during training
4. **Better generalization** to unseen motions

**Trade-offs:**
1. **Implementation complexity**: Requires complex-valued layers
2. **Debugging difficulty**: Less intuitive than real-valued networks
3. **Limited tool support**: Fewer visualization tools for complex activations

## Advanced Topics

### 1. Complex Dropout

For regularization in complex networks:

```python
class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, real, imag):
        if self.training:
            # Same mask for real and imaginary
            mask = torch.bernoulli(
                torch.ones_like(real) * (1 - self.p)
            )
            real = real * mask / (1 - self.p)
            imag = imag * mask / (1 - self.p)
        return real, imag
```

### 2. Complex Residual Connections

```python
class ComplexResidualBlock(nn.Module):
    def forward(self, real, imag):
        identity_r, identity_i = real, imag
        
        # Process through layers
        real, imag = self.layers(real, imag)
        
        # Add residual connection
        real = real + identity_r
        imag = imag + identity_i
        
        return real, imag
```

### 3. Complex Attention Mechanisms

```python
class ComplexAttention(nn.Module):
    def forward(self, real, imag):
        # Compute complex magnitude as attention weights
        magnitude = torch.sqrt(real**2 + imag**2)
        attention = F.softmax(magnitude, dim=-1)
        
        # Apply attention
        real = real * attention
        imag = imag * attention
        
        return real, imag
```

## References

### Academic Papers

1. **Original PhaseNet**
   - Meyer & Djelouah, "PhaseNet for Video Frame Interpolation" (CVPR 2018)
   - [Paper](https://arxiv.org/abs/1804.00884)

2. **Complex-Valued Networks**
   - Trabelsi et al., "Deep Complex Networks" (ICLR 2018)
   - [Paper](https://arxiv.org/abs/1705.09792)

3. **Steerable Pyramids**
   - Portilla & Simoncelli, "A Parametric Texture Model..." (IJCV 2000)
   - Simoncelli & Freeman, "The Steerable Pyramid..." (ICASSP 1995)

4. **Complex RNNs**
   - Arjovsky et al., "Unitary Evolution RNNs" (ICML 2016)
   - [Paper](https://arxiv.org/abs/1511.06464)

### Implementations

- Original PhaseNet: [GitHub](https://github.com/csjcai/PhaseNet)
- PyTorch Steerable Pyramid: [GitHub](https://github.com/tomrunia/PyTorchSteerablePyramid)
- Complex PyTorch: [GitHub](https://github.com/wavefrontshaping/complexPyTorch)

## Contributing

Contributions are welcome! Areas for improvement:

1. **Complex RNN layers** for temporal modeling
2. **Complex attention mechanisms** for feature weighting
3. **Uncertainty estimation** using complex distributions
4. **Multi-scale training** strategies
5. **Quantitative evaluation** on standard benchmarks

## License

This project extends the original PhaseNet implementation and is provided under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{meyer2018phasenet,
  title={PhaseNet for Video Frame Interpolation},
  author={Meyer, Simone and Djelouah, Abdelaziz and McWilliams, Brian and Sorkine-Hornung, Alexander and Gross, Markus and Schroers, Christopher},
  booktitle={CVPR},
  year={2018}
}

@inproceedings{trabelsi2018deep,
  title={Deep Complex Networks},
  author={Trabelsi, Chiheb and Bilaniuk, Olexa and Zhang, Ying and Serdyuk, Dmitriy and Subramanian, Sandeep and Santos, Jo{\~a}o Felipe and Mehri, Soroush and Rostamzadeh, Negar and Bengio, Yoshua and Pal, Christopher J},
  booktitle={ICLR},
  year={2018}
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository.
