# Understanding Adversarial Attacks: Theory and Implementation

## Introduction to Adversarial Attacks

Adversarial attacks exploit the vulnerability of deep neural networks to carefully crafted perturbations. These perturbations are designed to be imperceptible to humans but can cause classifiers to make incorrect predictions with high confidence. This document explains the mathematical foundations of different adversarial attack methods implemented in the project.

## Mathematical Foundations

### Neural Network Vulnerability

Consider a neural network classifier $f(x)$ that maps an input image $x$ to class probabilities. The network is vulnerable if there exists a small perturbation $\delta$ such that:
- $f(x) \neq f(x + \delta)$ (misclassification)
- $\|\delta\|_p \leq \varepsilon$ (perturbation is small under some norm)

Common norms used to measure perturbation size:
- $L_\infty$: Maximum absolute pixel change
- $L_2$: Euclidean distance
- $L_0$: Number of pixels changed

### Gradient-Based Attacks

Most adversarial attacks exploit the gradient of the loss function with respect to the input. Given:
- $x$: Original input
- $y$: True label
- $L(x, y)$: Loss function (e.g., cross-entropy)
- $\nabla_x L(x, y)$: Gradient of loss with respect to input

Attacks move in the direction that maximizes the loss.

## Attack Methods Explained

### Fast Gradient Sign Method (FGSM)

FGSM performs a single step in the direction of the gradient's sign:

$x_{adv} = x + \varepsilon \cdot \text{sign}(\nabla_x L(x, y))$

This creates a perturbation bounded by $\varepsilon$ in the $L_\infty$ norm.

#### FGSM Implementation Details

1. Forward pass to compute loss $L(x, y)$
2. Backward pass to compute gradient $\nabla_x L(x, y)$
3. Create perturbation by taking sign of gradient
4. Add perturbation to original image
5. Clip to valid image range

```python
def fgsm_attack(model, images, labels, epsilon):
    # Forward pass
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Create perturbation
    perturbation = epsilon * torch.sign(images.grad)
    
    # Create adversarial example
    adversarial = images + perturbation
    
    # Clip to valid range
    adversarial = torch.clamp(adversarial, min_val, max_val)
    
    return adversarial
```

### Projected Gradient Descent (PGD)

PGD is an iterative extension of FGSM, performing multiple steps and projecting back onto the $\varepsilon$-ball:

$x_0 = x + \text{uniform}(-\varepsilon, \varepsilon)$
$x_{t+1} = \Pi_{x, \varepsilon} \left( x_t + \alpha \cdot \text{sign}(\nabla_x L(x_t, y)) \right)$

where $\Pi_{x, \varepsilon}$ is the projection operation that ensures $\|x_{t+1} - x\|_\infty \leq \varepsilon$.

#### PGD Implementation Details

1. Start with random noise within the $\varepsilon$-ball
2. For each iteration:
   - Compute loss and gradient
   - Take a small step in the gradient sign direction
   - Project back onto the $\varepsilon$-ball around the original image
   - Clip to valid image range

```python
def pgd_attack(model, images, labels, epsilon, alpha, iterations):
    # Random start
    adversarial = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adversarial = torch.clamp(adversarial, min_val, max_val)
    
    for i in range(iterations):
        adversarial.requires_grad = True
        outputs = model(adversarial)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update
        adversarial = adversarial + alpha * torch.sign(adversarial.grad)
        
        # Project back to epsilon ball
        delta = torch.clamp(adversarial - images, -epsilon, epsilon)
        adversarial = images + delta
        
        # Clip to valid range
        adversarial = torch.clamp(adversarial, min_val, max_val)
        adversarial.grad.zero_()
    
    return adversarial
```

### Patch Attacks

Patch attacks modify only a small region of the image instead of adding perturbations to the entire image:

1. Choose a random location for a patch
2. Create a binary mask $M$ where patch area is 1, elsewhere 0
3. Optimize the patch to cause misclassification:
   $x_{adv} = x \odot (1-M) + \delta \odot M$
   where $\delta$ is the patch content and $\odot$ is element-wise multiplication

#### Patch Attack Implementation Details

1. Generate random patch locations for each image
2. Create binary masks for the patches
3. Initialize patches with random values
4. Optimize patch content through gradient descent
5. Apply only to the selected patch areas

```python
def patch_attack(model, images, labels, epsilon, patch_size, iterations):
    # Generate random patch locations
    patch_x = torch.randint(0, w - patch_size, (batch_size,))
    patch_y = torch.randint(0, h - patch_size, (batch_size,))
    
    # Create masks
    mask = torch.zeros_like(images)
    for i in range(batch_size):
        mask[i, :, patch_y[i]:patch_y[i]+patch_size, patch_x[i]:patch_x[i]+patch_size] = 1
    
    # Initialize with random patch
    adversarial = images.clone()
    patch = torch.empty_like(images).uniform_(-epsilon, epsilon)
    adversarial = images * (1-mask) + (images + patch) * mask
    
    # Optimize patch
    for i in range(iterations):
        adversarial.requires_grad = True
        outputs = model(adversarial)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Update only patch area
        adversarial = adversarial - alpha * adversarial.grad * mask
        delta = torch.clamp(adversarial - images, -epsilon, epsilon)
        adversarial = images + delta * mask
        adversarial = torch.clamp(adversarial, min_val, max_val)
        adversarial.grad.zero_()
    
    return adversarial
```

## Attack Transferability

Adversarial examples often transfer between models, meaning an example crafted to fool one model may also fool another. This is because:

1. Different models learn similar decision boundaries
2. Gradient directions often align across models
3. Universal adversarial perturbations exist

### Transferability Metrics

Transferability can be measured by the ratio of accuracy drop:

$\text{Transfer Rate} = \frac{\text{Accuracy Drop on Target Model}}{\text{Accuracy Drop on Source Model}}$

A higher transfer rate means attacks transfer more effectively.

## Mitigation Strategies

### Adversarial Training

Improve robustness by including adversarial examples in training:

$\min_\theta \mathbb{E}_{(x,y)} \left[ \max_{\|\delta\| \leq \varepsilon} L(x + \delta, y; \theta) \right]$

### Input Preprocessing

Apply transformations to inputs that preserve classification but disrupt adversarial perturbations:
- JPEG compression
- Bit depth reduction
- Gaussian noise addition
- Median filtering
- Feature squeezing

### Gradient Masking

Make gradients less useful for attackers by:
- Defensive distillation
- Gradient regularization
- Randomized models

### Certified Defenses

Provide provable guarantees against adversarial examples with bounded perturbations:
- Randomized smoothing
- Convex relaxations
- Interval bound propagation

## Conclusion

Adversarial attacks expose fundamental vulnerabilities in deep neural networks. Understanding these attacks helps develop more robust models and provides insights into how neural networks process information. The techniques implemented in this project demonstrate how small, carefully crafted perturbations can dramatically affect model performance.

## References

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.
2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083.
3. Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy (SP).
4. Athalye, A., Carlini, N., & Wagner, D. (2018). Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. arXiv preprint arXiv:1802.00420.
5. Brown, T. B., Mané, D., Roy, A., Abadi, M., & Gilmer, J. (2017). Adversarial patch. arXiv preprint arXiv:1712.09665.