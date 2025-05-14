# Adversarial Attacks on Deep Image Classifiers

This repository contains a comprehensive implementation of adversarial attacks on ResNet-34 and other ImageNet classifiers. The project demonstrates how small perturbations to images can cause deep learning models to misclassify them, while remaining visually similar to humans.

## Repository Contents

- `adversarial_attacks_project.py`: Main implementation script that generates adversarial examples using FGSM, PGD, and patch attacks
- `analysis_script.py`: Script for analyzing the generated adversarial datasets and their transferability
- `adversarial_theory.md`: Comprehensive explanation of the theory behind adversarial attacks

## Getting Started

### Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- matplotlib
- numpy
- Pillow
- tqdm

Install dependencies:
```bash
pip install torch torchvision matplotlib numpy pillow tqdm
```

### Dataset

1. Download the test dataset
2. Place it in a directory named `TestDataSet`
3. Make sure the `imagenet_class_index.json` file is in the same directory as the scripts

### Running the Code

1. Generate adversarial examples:
   ```bash
   python adversarial_attacks_project.py
   ```

2. Analyze the results:
   ```bash
   python analysis_script.py
   ```

## Project Structure

### Task 1: Baseline Evaluation
Evaluates a pre-trained ResNet-34 model on the clean test dataset.

### Task 2: FGSM Attack
Implements the Fast Gradient Sign Method to create adversarial examples with ε=0.02.

### Task 3: PGD Attack
Implements an improved attack using Projected Gradient Descent.

### Task 4: Patch Attack
Implements a localized attack that only modifies a small 32×32 patch of each image.

### Task 5: Transferability Analysis
Tests how well adversarial examples created for one model transfer to other models.

## Output

The scripts will generate:
- Three adversarial datasets in the `AdversarialSets` directory
- Visualizations of successful attacks
- Accuracy statistics and comparison plots
- Transferability analysis

## Project Files Explanation

### adversarial_attacks_project.py
The main implementation file that:
- Loads the pre-trained ResNet-34 model
- Calculates baseline accuracy
- Implements FGSM, PGD, and patch attacks
- Saves adversarial examples
- Evaluates attack effectiveness
- Conducts transferability analysis

### analysis_script.py
A supplementary script that:
- Analyzes perturbation metrics (L∞, L2, L0)
- Creates visualizations of successful attacks
- Analyzes transferability between models
- Generates confidence distribution plots
- Creates confusion matrices for misclassified examples


### adversarial_theory.md
A detailed explanation of:
- The mathematical foundations of adversarial attacks
- How different attack methods work
- Implementation details for each attack
- Attack transferability concepts
- Mitigation strategies

This project is provided for educational purposes only.

## Acknowledgments
- The project is based on the assignment from [Your Course]
- Uses the ImageNet-1K dataset for evaluation
- Builds on techniques from seminal papers on adversarial examples
