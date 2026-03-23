# JetFighter - Automated detection of Accessible and Problematic Visualizations


## Introduction

### The Accessibility Gap
Despite increasing awareness of accessibility in scientific communication, many biomedical research papers still contain images that are difficult or even impossible to interpret for people with **Color Vision Deficiency (CVD)**, which affects approximately 8% of men and 0.5% of women.

### The Problem with "Rainbow" Colormaps
A major issue is the widespread use of "rainbow" colormaps (like Jet). These maps are problematic because:
- **They are not perceptually uniform:** The human eye perceives changes in hue non-linearly, making small data variations in some regions look huge, while others disappear.
- **They create "Mach bands":** False edges appear where there are none in the data.
- **They lack luminance order:** When printed in black and white, a rainbow map becomes a confusing mess of grays, making high and low values indistinguishable.

### The Solution: Perceptual Uniformity
To address this, current visualization standards advocate for **perceptually uniform colormaps** (such as Viridis, Cividis, Magma, Inferno, or Plasma). In these maps, equal steps in data values correspond directly to equal perceptual steps in color. They also maintain a coherent luminance ramp (light to dark), ensuring the data remains interpretable even when printed in grayscale or viewed by individuals with total color blindness (monochromacy).

**JetFighter** is a machine learning-based tool designed to automatically screen PDF manuscripts. It detects figures, classifies their colormap usage, and flags accessibility issues.



## How it Works

The pipeline consists of three main steps:

### 1. Figure Detection
We use a **YOLO** object detection model to scan PDF pages and locate scientific figures, separating them from text and captions.

### 2. Classification
Once a figure is found, we extract its color histogram. A **Multi-Layer Perceptron (MLP)** classifier analyzes this histogram to determine if the figure uses a **Rainbow** colormap, a **Safe** gradient (a uniform perceptual colormap), or **Discrete** figures (like a bar chart).

### 3. Contrast Analysis
For figures classified as "Discrete," we perform an additional contrast check. We simulate how the figure looks in grayscale to ensure that different data categories have enough **luminance contrast** to remain distinguishable.

---

## Model Performance

### 1. Detector Performance (YOLO)

Our figure detector achieves high precision, ensuring that very few non-figures are incorrectly flagged.

| Metric | Value |
| :--- | :--- |
| **Precision** | 97.4% |
| **Recall** | 93.8% |
| **mAP50** | 98.0% |




#### Confusion Matrix:

| | Predicted Figure | Predicted Background |
| :--- | :---: | :---: |
| **True Figure** | **63** | 11 |
| **True Background** | 1 | - |


### 2. Classifier Performance (Histogram MLP)

The classifier was evaluated on a  validation set of **105 images** (35 per class).

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Rainbow (0)** | 0.88 | 1.00 | 0.93 | 35 |
| **Safe (1)** | 0.92 | 1.00 | 0.96 | 35 |
| **Discrete (2)** | 1.00 | 0.77 | 0.87 | 35 |
| **Overall Accuracy** | | | **92.4%** | 105 |

#### Confusion Matrix:

| | Pred: Rainbow | Pred: Safe | Pred: Discrete |
| :--- | :---: | :---: | :---: |
| **True Rainbow** | **35** | 0 | 0 |
| **True Safe** | 0 | **35** | 0 |
| **True Discrete** | 5 | 3 | **27** |


---

## The Web Application

We provide a user-friendly web interface for researchers to verify their own manuscripts.

1. **Upload**: Drag and drop your PDF manuscript.
2. **Analysis**: The backend processes every page.
3. **Result**: Each figure is assigned one of four categories:

| Category | Status | Meaning |
| :--- | :--- | :--- |
| **safe_gradient** | 🟢 Safe | Uses an accessible, perceptually uniform colormap. |
| **accessible_discrete** | 🟢 Safe | Discrete data with sufficient luminance contrast. |
| **rainbow_gradient** | 🔴 Problematic | Uses a distorted rainbow/jet map. |
| **problematic_discrete** | 🔴 Problematic | Discrete colors are indistinguishable in grayscale. |

This immediate feedback helps authors ensure their work is accessible to the widest possible audience.

**Looking for the technical documentation and installation guide?**  
-> Check the [Technical README](jetfighter-monorepo/README.md) in the monorepo folder.
