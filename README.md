# 3D Shape Retrieval System

A multimedia retrieval system that lets you browse, analyze, and search for similar 3D shapes. The system standardizes shapes and uses comparison methods to find similar objects.
---


## Required Libraries

### Main Application Dependencies

```
# Core Python Libraries
numpy                # Numerical computations and arrays
pandas               # Data manipulation and CSV handling
matplotlib           # Plotting and visualization

# 3D Processing
trimesh              # 3D mesh loading and processing
pymeshlab            # Advanced mesh operations (remeshing, cleaning)
vedo                 # 3D visualization and rendering

# GUI Framework
PyQt6                # Graphical user interface
vtkmodules           # VTK rendering integration with Qt

# Machine Learning
scikit-learn         # KNN search, t-SNE, metrics, standardization

# Standard Libraries (included with Python)
os, sys, csv, math, hashlib, collections, re
typing, concurrent.futures
```

## Setup and Usage

To set up the environment and run the application, please follow the instructions below.

### For Windows Command Prompt (`cmd.exe`):

1. **Navigate to the correct project folder**
    ```cmd
    cd Foldername
    ```

3.  **Create a virtual environment:**
    ```cmd
    python -m venv venv
    ```

4.  **Activate the virtual environment:**
    ```cmd
    venv\Scripts\activate.bat
    ```

5.  **Install the required packages:**
    ```cmd
    pip install numpy pandas matplotlib trimesh pymeshlab vedo PyQt6 scikit-learn vtk scipy
    ```

6.  **Run the application:**
    ```cmd
    python Main.py
    ```

This opens a graphical interface where you can:
- Browse 3D shapes by category
- View shapes in 3D
- See detailed shape features
- Search for similar shapes

---

## Main Files Overview

### **Main.py** - The Main Application
The primary interface for the entire system.

**What it does:**
- Shows a 3D viewer with controls on the left and feature displays on the right
- Lets you select a category (like "Car", "Guitar", "Bird") and then pick a specific object
- Displays the shape in 3D - you can see both the original and a normalized (standardized) version
- Shows shape features: histograms (D1, D2, D3, D4, A3) and scalar values (sphericity, convexity, etc.)
- Searches for similar shapes using different comparison methods
- Shows the top 5 most similar shapes with their dissimilarity scores

**How to use:**
1. Select a category from the dropdown
2. Click on an object in the list
3. Wait for the shape to load and normalize (this takes a few seconds)
4. Toggle "Show Normalized" to see the standardized version
5. Choose a search metric (like "manhattan" or "euclidean")
6. Click "Find Similar" to see the 5 most similar shapes

---

### **RemeshAndNormalise.py** - Shape Preparation
Prepares shapes so they can be compared fairly.

**What it does:**
- **Remeshing**: Adjusts every shape to have exactly 5000 vertices (points)
  - Too many vertices? It reduces them using decimation
  - Too few vertices? It adds more using subdivision
- **Normalization**: Standardizes shape position and size
  - Centers the shape at the origin point (0, 0, 0)
  - Scales it to fit in a unit cube (size = 1)
- **Flipping**: Ensures shapes face the same direction using PCA (Principal Component Analysis)


---

### **feature_extraction.py** - Shape Analysis
Converts 3D shapes into numbers that can be compared.

**What it does:**
- **Histogram Features** (5 types, 20 bins each = 100 values):
  - **D1**: Distance from random points to the center (shows compactness)
  - **D2**: Distance between pairs of random points (shows overall size distribution)
  - **D3**: Square root of triangle areas (shows surface detail)
  - **D4**: Cube root of tetrahedron volumes (shows 3D space filling)
  - **A3**: Angles between random point triplets (shows angular properties)

- **Scalar Features** (6 single values):
  - **Surface area**: Total surface size
  - **Sphericity**: How sphere-like the shape is (1 = perfect sphere)
  - **Rectangularity**: How well it fits in a box
  - **Diameter**: Maximum distance across the shape
  - **Convexity**: Ratio of shape volume to its convex hull
  - **Eccentricity**: How elongated the shape is

**Output:**
Creates a CSV file with all features for all shapes (111 numbers per shape).

---

### **Querying.py** - Shape Search Engine
Finds similar shapes by comparing their features.

**What it does:**
- Loads the feature database (CSV file with all shape features)
- Compares a query shape to all other shapes using different distance metrics
- Ranks shapes from most similar (smallest distance) to least similar

**Distance Metrics Available:**
- **Euclidean**: Straight-line distance (like measuring with a ruler)
- **Manhattan**: Grid-based distance (like walking city blocks)
- **Manhattan + Chi-squared**: Manhattan with special handling for histograms
- **Manhattan + EMD**: Manhattan with Earth Mover's Distance for histograms
- **Manhattan + Kullback-Leibler**: Manhattan with KL divergence for histograms
- **KNN**: K-Nearest Neighbors using scikit-learn

**Feature Weighting:**
Different features have different importance. For example, A3 (angles) might be weighted 9.0 while D4 might be weighted 0.5. This lets important features have more influence on the search results.

---

### **Evaluation.py** - Performance Measurement
Measures how well the search system works.

**What it does:**
- Runs queries and checks if the returned shapes are actually similar (same category)
- Calculates performance metrics:
  - **Precision**: Of the returned shapes, how many are correct?
  - **Recall**: Of all correct shapes, how many were found?
  - **F1 Score**: Balance between precision and recall
  - **MAP** (Mean Average Precision): Overall search quality
  - **AUC** (Area Under Curve): How well it separates similar from different shapes
  - **1st Tier Accuracy**: Percentage of correct shapes in the top results
- Creates ROC curves to visualize performance
- Ensures fair comparison by weighting all categories equally (small and large categories count the same)

**How it works:**
1. Takes query results from CSV files
2. For each query, checks if returned shapes are in the same category
3. Calculates metrics per category, then averages across all categories
4. Generates plots and statistics

---

### **extract_all_query_answers.py** - Batch Testing
Tests the entire database by using every shape as a query.

**What it does:**
- Loops through all shapes in the database
- Uses each shape as a query to find similar shapes
- Tests multiple distance metrics at once
- Saves results to CSV files for evaluation

**Output:**
Creates files like `results_neutral.csv` with columns:
- Query information (category, object name)
- Retrieved categories at each rank (rank_1, rank_2, etc.)
- Distance values for each result

**Why this is useful:**
You need lots of test data to measure how well the system works. This automates testing thousands of queries.

---

### **scalability.py** - Data Visualization
Creates 2D maps of the shape database.

**What it does:**
- Takes high-dimensional feature data (111 numbers per shape)
- Reduces it to 2D using t-SNE (a dimension reduction algorithm)
- Creates scatter plots where each dot is a shape
- Colors dots by category (all cars are one color, all guitars another, etc.)

**Why this is useful:**
You can visually see if similar shapes cluster together. Good features will make all cars appear close to each other, separated from guitars, etc.

---

### **tsne_GUI.py** - Interactive Visualization Viewer
An interactive tool to explore the t-SNE visualization.

**What it does:**
- Shows the t-SNE scatter plot in a window
- Lets you toggle categories on/off to focus on specific shapes
- Hover over points to see which shape they represent
- Highlight categories to see them more clearly

**How to use:**
1. Run `python tsne_GUI.py`
2. Use checkboxes to show/hide categories
3. Hover over dots to see shape names
4. Click on categories to highlight them

---
## Folder Structure

```
Multimedia_Retrieval/
├── Main.py                          # Main application (START HERE)
├── RemeshAndNormalise.py            # Shape preprocessing
├── feature_extraction.py            # Feature extraction
├── Querying.py                      # Search engine
├── Evaluation.py                    # Performance metrics
├── extract_all_query_answers.py    # Batch testing
├── scalability.py                   # t-SNE visualization
├── tsne_GUI.py                      # Interactive viewer
│
├── ShapeDatabase_INFOMR-master/     # Original 3D shapes
│   └── Original Database/
│       ├── Car/
│       ├── Guitar/
│       └── ...
│
├── Normalised-objects/              # Processed shapes (5000 vertices each)
│   ├── Car/
│   ├── Guitar/
│   └── ...
│
├── Feature-matrix/                  # Feature database
│   └── all_features.csv            # 111 features per shape
│
├── QueryResults/                    # Search test results
│   ├── results_neutral.csv
│   └── results_weighted.csv
│
└── plots/                          # Generated visualizations
```





_This overview was made with the help of AI, The content was checked and adjusted by the creators of this projects to make sure all content is accurate_
