#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Attack Analysis Script
This script loads and analyzes saved adversarial datasets.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Define normalization parameters (ImageNet standard)
mean_norms = np.array([0.485, 0.456, 0.406])
std_norms = np.array([0.229, 0.224, 0.225])

# Create transforms
plain_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_norms, std=std_norms)
])

# Path settings
dataset_path = "./TestDataSet"  # Path to original test dataset
adv_path = "./AdversarialSets"  # Path to saved adversarial datasets
results_path = "./Analysis"     # Path to save analysis results
os.makedirs(results_path, exist_ok=True)

# Load class labels
with open("imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)
    
# Create a mapping from ImageNet indices to class names
idx_to_class = {int(k): v[1] for k, v in class_idx.items()}

def load_model(model_name):
    """Load a pre-trained model."""
    if model_name == "resnet34":
        model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    elif model_name == "densenet121":
        model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
    elif model_name == "vgg16":
        model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
    elif model_name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    model.eval()
    return model

def denormalize(tensor):
    """Convert normalized image tensor to displayable image."""
    tensor = tensor.clone().detach().to('cpu').numpy().transpose(1, 2, 0)
    tensor = tensor * std_norms[None, None, :] + mean_norms[None, None, :]
    tensor = np.clip(tensor, 0, 1)
    return tensor

def calculate_topk_accuracy(model, dataloader, k=5):
    """Calculate top-k accuracy."""
    model.eval()
    correct_top1 = 0
    correct_topk = 0
    total = 0
    
    pred_classes = []
    true_classes = []
    confidences = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Calculating Top-{k} Accuracy"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Top-1 Accuracy
            _, predicted = outputs.max(1)
            correct_top1 += (predicted == labels).sum().item()
            
            # Top-k Accuracy
            _, topk_predicted = outputs.topk(k, dim=1)
            for i in range(labels.size(0)):
                correct_topk += (labels[i] in topk_predicted[i]).sum().item()
            
            # Store predictions
            pred_classes.extend(predicted.cpu().numpy())
            true_classes.extend(labels.cpu().numpy())
            confidences.extend(probs.max(dim=1)[0].cpu().numpy())
            
            total += labels.size(0)
    
    top1_accuracy = correct_top1 / total
    topk_accuracy = correct_topk / total
    
    return top1_accuracy, topk_accuracy, pred_classes, true_classes, confidences

def visualize_accuracy_comparison(models, dataset_names, results):
    """Visualize accuracy comparison across models and datasets."""
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(dataset_names))
    width = 0.8 / len(models)
    
    for i, model_name in enumerate(models):
        top1_values = [results[model_name][dataset]['top1'] for dataset in dataset_names]
        plt.bar(x + i*width - width*(len(models)-1)/2, top1_values, width, label=model_name)
    
    plt.xlabel('Dataset')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Model Performance Comparison Across Datasets')
    plt.xticks(x, [name.split('_')[-1] for name in dataset_names], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(results_path, "accuracy_comparison.png"))
    plt.close()

def visualize_confusion_matrix(true_classes, pred_classes, dataset_name, model_name):
    """Create a visualization of most confused classes."""
    # Count occurrences of each true-predicted class pair
    confusion_counts = {}
    for true, pred in zip(true_classes, pred_classes):
        if true == pred:
            continue  # Skip correct predictions
        
        key = (true, pred)
        if key in confusion_counts:
            confusion_counts[key] += 1
        else:
            confusion_counts[key] = 1
    
    # Sort by frequency
    sorted_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 10
    top_confusions = sorted_confusions[:10]
    
    if not top_confusions:
        print(f"No misclassifications found for {model_name} on {dataset_name}")
        return
    
    # Prepare data for plotting
    labels = [f"{idx_to_class[true]} → {idx_to_class[pred]}" for (true, pred), _ in top_confusions]
    counts = [count for _, count in top_confusions]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(labels)), counts, color='coral')
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Count')
    plt.title(f'Top Misclassifications: {model_name} on {dataset_name}')
    plt.gca().invert_yaxis()  # Highest count at the top
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(results_path, f"confusion_{model_name}_{dataset_name}.png"))
    plt.close()

def analyze_confidence_distribution(confidences, dataset_name, model_name):
    """Analyze and visualize the confidence distribution."""
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.hist(confidences, bins=20, alpha=0.7, color='blue')
    plt.axvline(np.mean(confidences), color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {np.mean(confidences):.3f}')
    
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title(f'Confidence Distribution: {model_name} on {dataset_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(results_path, f"confidence_{model_name}_{dataset_name}.png"))
    plt.close()

def calculate_perturbation_metrics(original_dataset, adversarial_dataset, name):
    """Calculate metrics about the perturbations."""
    # Load a batch of images
    orig_loader = DataLoader(original_dataset, batch_size=100, shuffle=False)
    adv_loader = DataLoader(adversarial_dataset, batch_size=100, shuffle=False)
    
    # Get first batch
    orig_images, _ = next(iter(orig_loader))
    adv_images, _ = next(iter(adv_loader))
    
    # Calculate L-infinity distance
    l_inf = (adv_images - orig_images).abs().max(dim=3)[0].max(dim=2)[0].max(dim=1)[0]
    avg_l_inf = l_inf.mean().item()
    max_l_inf = l_inf.max().item()
    
    # Calculate L2 distance
    l2 = ((adv_images - orig_images)**2).sum(dim=(1,2,3)).sqrt()
    avg_l2 = l2.mean().item()
    max_l2 = l2.max().item()
    
    # Calculate L0 (number of changed pixels)
    l0 = ((adv_images - orig_images).abs() > 0.01).sum(dim=(1,2,3))
    avg_l0 = l0.float().mean().item()
    max_l0 = l0.max().item()
    
    # Print results
    print(f"\nPerturbation Metrics for {name}:")
    print(f"  L∞ Distance - Average: {avg_l_inf:.4f}, Max: {max_l_inf:.4f}")
    print(f"  L2 Distance - Average: {avg_l2:.4f}, Max: {max_l2:.4f}")
    print(f"  L0 Distance - Average: {avg_l0:.1f}, Max: {max_l0}")
    
    # Return metrics
    return {
        'l_inf': {'avg': avg_l_inf, 'max': max_l_inf},
        'l2': {'avg': avg_l2, 'max': max_l2},
        'l0': {'avg': avg_l0, 'max': max_l0}
    }

def visualize_perturbation_comparison(metrics, dataset_names):
    """Visualize perturbation metrics across datasets."""
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(dataset_names)-1)  # Skip original dataset
    width = 0.3
    
    # Only plot for adversarial datasets (skip original)
    datasets = dataset_names[1:]
    
    # Plot L-infinity
    l_inf_values = [metrics[dataset]['l_inf']['avg'] for dataset in datasets]
    plt.bar(x - width, l_inf_values, width, label='L∞ Distance')
    
    # Plot L2
    l2_values = [metrics[dataset]['l2']['avg'] / 10 for dataset in datasets]  # Scale for visibility
    plt.bar(x, l2_values, width, label='L2 Distance (÷10)')
    
    # Plot L0
    l0_values = [metrics[dataset]['l0']['avg'] / 1000 for dataset in datasets]  # Scale for visibility
    plt.bar(x + width, l0_values, width, label='L0 Distance (÷1000)')
    
    plt.xlabel('Dataset')
    plt.ylabel('Distance Metric')
    plt.title('Perturbation Comparison Across Adversarial Datasets')
    plt.xticks(x, [name.split('_')[-1] for name in datasets])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(results_path, "perturbation_comparison.png"))
    plt.close()

def visualize_adv_examples(original_dataset, adversarial_datasets, dataset_names, models):
    """Visualize examples from each dataset and their predictions."""
    # Select a few random indices
    indices = np.random.choice(len(original_dataset), 3, replace=False)
    
    # Load models
    model_objects = {name: load_model(name) for name in models}
    
    # Create a large figure with a grid of images
    fig, axes = plt.subplots(len(indices), len(dataset_names), figsize=(4*len(dataset_names), 4*len(indices)))
    
    # Get images and predictions
    for i, idx in enumerate(indices):
        # Get original image
        original_image, original_label = original_dataset[idx]
        original_image = original_image.unsqueeze(0).to(device)
        
        # Get predictions from each model
        original_preds = {}
        for model_name, model in model_objects.items():
            with torch.no_grad():
                outputs = model(original_image)
                _, pred = outputs.max(1)
                original_preds[model_name] = pred.item()
        
        # Display original image
        ax = axes[i, 0]
        ax.imshow(denormalize(original_dataset[idx][0]))
        ax.set_title(f"Original\nTrue: {idx_to_class[original_label]}\nResNet: {idx_to_class[original_preds['resnet34']]}")
        ax.axis('off')
        
        # Display adversarial images
        for j, (dataset, name) in enumerate(zip(adversarial_datasets, dataset_names[1:])):
            if j >= len(dataset_names) - 1:
                continue
                
            # Get adversarial image
            adv_image, adv_label = dataset[idx]
            adv_image_tensor = adv_image.unsqueeze(0).to(device)
            
            # Get predictions from each model
            adv_preds = {}
            for model_name, model in model_objects.items():
                with torch.no_grad():
                    outputs = model(adv_image_tensor)
                    _, pred = outputs.max(1)
                    adv_preds[model_name] = pred.item()
            
            # Display adversarial image
            ax = axes[i, j+1]
            ax.imshow(denormalize(adv_image))
            ax.set_title(f"{name.split('_')[-1]}\nTrue: {idx_to_class[adv_label]}\nResNet: {idx_to_class[adv_preds['resnet34']]}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "adversarial_examples.png"))
    plt.close()

def analyze_transferability(results, dataset_names, models):
    """Analyze and visualize attack transferability."""
    # Calculate transfer rates
    transfer_rates = {}
    baseline_dataset = dataset_names[0]  # Original dataset
    
    for dataset in dataset_names[1:]:  # Skip original dataset
        transfer_rates[dataset] = {}
        
        for source_model in models:
            source_drop = results[source_model][baseline_dataset]['top1'] - results[source_model][dataset]['top1']
            
            for target_model in models:
                if source_model == target_model:
                    continue
                
                target_drop = results[target_model][baseline_dataset]['top1'] - results[target_model][dataset]['top1']
                
                if source_drop > 0:
                    transfer_rates[dataset][(source_model, target_model)] = target_drop / source_drop
                else:
                    transfer_rates[dataset][(source_model, target_model)] = 0
    
    # Create transfer rate heatmap for each dataset
    for dataset in dataset_names[1:]:
        plt.figure(figsize=(10, 8))
        
        # Prepare data for heatmap
        heatmap_data = np.zeros((len(models), len(models)))
        
        for i, source in enumerate(models):
            for j, target in enumerate(models):
                if source == target:
                    heatmap_data[i, j] = 1.0  # Perfect self-transfer
                else:
                    heatmap_data[i, j] = transfer_rates[dataset].get((source, target), 0) * 100
        
        # Create heatmap
        plt.imshow(heatmap_data, cmap='YlOrRd', vmin=0, vmax=100)
        plt.colorbar(label='Transfer Rate (%)')
        
        # Add labels
        plt.xticks(range(len(models)), models, rotation=45)
        plt.yticks(range(len(models)), models)
        plt.xlabel('Target Model')
        plt.ylabel('Source Model')
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(models)):
                text_color = 'black' if heatmap_data[i, j] < 70 else 'white'
                plt.text(j, i, f"{heatmap_data[i, j]:.1f}%", 
                         ha="center", va="center", color=text_color)
        
        plt.title(f'Attack Transferability: {dataset.split("_")[-1]}')
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f"transferability_{dataset}.png"))
        plt.close()

# Main analysis function
def main():
    print("Starting adversarial attack analysis...")
    
    # Load the datasets
    original_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=plain_transforms)
    
    # Find adversarial dataset directories
    adv_datasets = []
    dataset_names = ["Original"]
    
    for folder in sorted(os.listdir(adv_path)):
        if os.path.isdir(os.path.join(adv_path, folder)) and folder.startswith("Adversarial"):
            print(f"Found adversarial dataset: {folder}")
            adv_dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(adv_path, folder), 
                transform=plain_transforms
            )
            adv_datasets.append(adv_dataset)
            dataset_names.append(folder)
    
    # Create data loaders
    batch_size = 16
    original_loader = DataLoader(original_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    adv_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2) 
                  for dataset in adv_datasets]
    
    # Models to evaluate
    models = ["resnet34", "densenet121", "vgg16", "efficientnet_b0"]
    
    # Store results
    results = {model_name: {} for model_name in models}
    
    # Calculate perturbation metrics
    perturbation_metrics = {}
    for adv_dataset, name in zip(adv_datasets, dataset_names[1:]):
        perturbation_metrics[name] = calculate_perturbation_metrics(original_dataset, adv_dataset, name)
    
    # Visualize perturbation comparison
    visualize_perturbation_comparison(perturbation_metrics, dataset_names)
    
    # Evaluate each model on each dataset
    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        model = load_model(model_name)
        
        # Evaluate on original dataset
        print(f"  Evaluating on original dataset...")
        top1, top5, pred_classes, true_classes, confidences = calculate_topk_accuracy(model, original_loader)
        results[model_name]["Original"] = {'top1': top1 * 100, 'top5': top5 * 100}
        print(f"  Original - Top-1: {top1*100:.2f}%, Top-5: {top5*100:.2f}%")
        
        # Visualize confusion matrix for original dataset
        visualize_confusion_matrix(true_classes, pred_classes, "Original", model_name)
        
        # Analyze confidence distribution
        analyze_confidence_distribution(confidences, "Original", model_name)
        
        # Evaluate on adversarial datasets
        for loader, name in zip(adv_loaders, dataset_names[1:]):
            print(f"  Evaluating on {name}...")
            top1, top5, pred_classes, true_classes, confidences = calculate_topk_accuracy(model, loader)
            results[model_name][name] = {'top1': top1 * 100, 'top5': top5 * 100}
            print(f"  {name} - Top-1: {top1*100:.2f}%, Top-5: {top5*100:.2f}%")
            
            # Visualize confusion matrix
            visualize_confusion_matrix(true_classes, pred_classes, name, model_name)
            
            # Analyze confidence distribution
            analyze_confidence_distribution(confidences, name, model_name)
    
    # Visualize accuracy comparison
    visualize_accuracy_comparison(models, dataset_names, results)
    
    # Visualize adversarial examples
    visualize_adv_examples(original_dataset, adv_datasets, dataset_names, models)
    
    # Analyze transferability
    analyze_transferability(results, dataset_names, models)
    
    print("\nAnalysis complete! Results saved to:", results_path)

if __name__ == "__main__":
    main()
