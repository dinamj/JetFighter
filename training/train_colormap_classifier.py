"""
JetFighter MLP Training
-> trains an MLP on 3D color histograms instead of image structure

Classes:
    0: rainbow_gradient (problematic colormaps like jet, turbo)
    1: safe_gradient    (accessible colormaps like viridis, plasma)
    2: discrete         (discrete colors, no gradients)

Usage:
    python training/train_histogram_classifier.py
    python training/train_histogram_classifier.py --epochs 30 --batch-size 128
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from argparse import ArgumentParser
from datetime import datetime
import json


def build_parser():
    parser = ArgumentParser(description='Train histogram-based colormap classifier')
    
    parser.add_argument('--data-dir',
                        dest='data_dir',
                        help='Data directory (default: data/jetfighter3)',
                        default='data/jetfighter3')
    
    parser.add_argument('--output-dir',
                        dest='output_dir',
                        help='Output directory for models (default: models)',
                        default='models')
    
    parser.add_argument('--epochs',
                        dest='epochs',
                        type=int,
                        help='Number of epochs (default: 100)',
                        default=100)
    
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        type=int,
                        help='Batch size (default: 16)',
                        default=16)
    
    parser.add_argument('--lr',
                        dest='learning_rate',
                        type=float,
                        help='Learning rate (default: 0.001)',
                        default=0.001)
    
    parser.add_argument('--bins',
                        dest='bins',
                        type=int,
                        help='Number of bins per color channel (default: 8 -> 8x8x8=512 features)',
                        default=8)
    
    parser.add_argument('--resize',
                        dest='resize',
                        type=int,
                        help='Resize images (0=original size, default: 1280)',
                        default=1280)
    
    return parser


class ColorHistogramDataset(Dataset):
    
    def __init__(self, root_dir, split='train', bins=8, resize=600):
        self.root_dir = Path(root_dir) / split
        self.bins = bins
        self.resize = resize
        self.image_paths = []
        self.labels = []
        
        if not self.root_dir.exists():
            raise ValueError(f"Directory not found: {self.root_dir}")
        
        print(f"Loading {split} data from {self.root_dir}...")
        
        # Load classes (folders should be named like 'class_0_rainbow')
        classes = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        for cls_dir in classes:
            try:
                # Parse label from folder name
                label = int(cls_dir.name.split('_')[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid class folder format: {cls_dir.name}")
            
            # Collect images
            img_count = 0
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for img_path in cls_dir.glob(ext):
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    img_count += 1
            
            print(f"Class {label}: {img_count} images")
        
        print(f"Total images: {len(self.image_paths)}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            features: Normalized flattened 3D histogram
            label: Class index
        """
        img_path = self.image_paths[idx]
        
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                
                # Resize if specified
                if self.resize and self.resize > 0:
                    img = img.resize((self.resize, self.resize))
                
                arr = np.array(img)
            
            # Compute 3D Histogram (R, G, B)
            hist, _ = np.histogramdd(
                arr.reshape(-1, 3), 
                bins=(self.bins, self.bins, self.bins), 
                range=((0, 256), (0, 256), (0, 256))
            )
            
            # Filter out pure white
            hist[self.bins-1, self.bins-1, self.bins-1] = 0.0  # white
            
            # Normalize to make it resolution-invariant
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist = hist / hist_sum
            
            # Flatten to vector
            features = torch.FloatTensor(hist.flatten())
            label = torch.LongTensor([self.labels[idx]])
            
            return features, label.squeeze()
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(self.bins ** 3), torch.tensor(0)


class ColorClassifier(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP).
    Input: Flattened histogram vector.
    Output: Class probabilities.
    """
    
    def __init__(self, input_dim=512, num_classes=3, hidden_dim=256, dropout=0.3):
        super(ColorClassifier, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)


def train_model(data_dir, output_dir, epochs=20, batch_size=16, 
                learning_rate=0.001, bins=8, resize=600):
    
    # Setup paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    project_path = output_path / 'histogram_training'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    print(f"Features: {bins}^3 = {bins**3} bins")
    
    # Load Datasets
    train_dataset = ColorHistogramDataset(data_dir, 'train', bins=bins, resize=resize)
    val_dataset = ColorHistogramDataset(data_dir, 'val', bins=bins, resize=resize)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model init
    input_dim = bins ** 3
    model = ColorClassifier(input_dim=input_dim, num_classes=3)
    model = model.to(device)
    
    # Compute Class Weights (to handle imbalanced data)
    labels = train_dataset.labels
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Inverse frequency weighting
    weights = total_samples / (len(class_counts) * class_counts) ** 0.5
    class_weights = torch.FloatTensor(weights).to(device)
    
    print(f"Class Weights: {weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    print("\n--- Starting Training ---")
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss {avg_loss:.4f}, Train Acc {train_acc:.1f}%, Val Acc {val_acc:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Final Evaluation
    print("\n--- Final Evaluation ---")
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    class_names = ['Rainbow (0)', 'Safe (1)', 'Discrete (2)']
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Save Outputs
    run_path = project_path / run_name
    weights_path = run_path / 'weights'
    weights_path.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model_state_dict': best_model_state,
        'config': {
            'bins': bins,
            'input_dim': input_dim,
            'num_classes': 3,
            'resize': resize
        },
        'metrics': {'val_accuracy': best_val_acc}
    }
    
    # Save best model for this run
    best_model_path = weights_path / 'best.pth'
    torch.save(model_data, best_model_path)
    print(f"\nSaved best run model: {best_model_path}")
    
    # Save metrics JSON
    metrics_path = run_path / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'val_accuracy': best_val_acc,
            'confusion_matrix': cm.tolist()
        }, f, indent=2)
    
    # Update global latest model
    latest_path = output_path / 'histogram_classifier.pth'
    torch.save(model_data, latest_path)
    print(f"Updated global model: {latest_path}")
    
    print("\nTraining Complete.")
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    return train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        bins=args.bins,
        resize=args.resize
    )


if __name__ == '__main__':
    exit(main())