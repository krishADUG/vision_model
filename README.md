# Vision Model

A computer vision project repository structured around reusable data pipelines, model components, and separate processing workflows for **2D vision** and **3D vision** tasks. The codebase is organized like a ML project, with dedicated folders for data, source code, notebooks, and tests.

## Overview

This repository is designed as a **multi-domain computer vision project**. It separates:

* raw and processed data
* training / execution entry points
* shared model definitions
* 2D image processing logic
* 3D vision processing logic
* testing infrastructure

That structure is strong for building scalable machine learning projects because it keeps experiments, data handling, and reusable source code cleanly separated.

## Repository Structure

```
vision_model/
├── data/
│   ├── processed/
│   └── raw/
├── main.py
├── notebooks/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py
│   ├── models/
│   │   └── model.py
│   ├── vision2d/
│   │   └── processor.py
│   └── vision3d/
│       └── processor.py
└── tests/
```

## Concept Roadmap

This repository is organized around the following idea:

1. **Project-oriented ML engineering** — separating code, data, and experiments
2. **Data pipeline design** — moving from raw inputs to processed inputs
3. **Reusable model architecture** — centralizing model-related logic in `src/models`
4. **2D computer vision workflows** — image-based preprocessing and inference pipelines
5. **3D computer vision workflows** — handling depth, volumetric, or spatial vision data
6. **Experimentation support** — notebooks for analysis and prototyping
7. **Software quality practices** — test folder for validation and maintainability

## Breakdown

### Root Files

#### `main.py`

This is the top-level execution script for the repository.

**Role:**

* starts the project pipeline
* loads configs or input paths
* calls processing or model code
* acts as the CLI-style entry point for running the project


#### `requirements.txt`

Defines the Python dependencies needed to run the project.

**Role:**

* lists required libraries
* helps recreate the environment
* supports reproducibility across machines

---

#### `data/raw/`

Stores the original, unmodified source data.

**Use:**

* raw images
* raw scans
* sensor captures
* source files before preprocessing

**Concept:**
Keeping raw data untouched preserves the original input and makes preprocessing reproducible.

#### `data/processed/`

Stores cleaned, transformed, resized, filtered, or feature-ready data.

**Use:**

* normalized images
* resized inputs
* extracted features
* serialized arrays or tensors

**Concept:**
This is the version of the data used more directly by the model or downstream processors.

---

### `notebooks/`

Used for interactive experimentation.

**Role:**

* exploratory data analysis
* model debugging
* quick visualization
* trying preprocessing ideas
* evaluating outputs interactively


---

### `src/`

The core source code directory.

#### `src/main.py`

A source-level entry point that coordinate internal modules more cleanly than the root `main.py`.

**Role:**

* orchestrates data loading and preprocessing
* connects processors and model logic
* supports modular execution from inside the package

---

#### `src/models/model.py`

The central model implementation in the repository.

**Role:**

* defines the architecture class
* contains forward-pass logic
* include model initialization helpers
* shared by both 2D and 3D workflows

* model abstraction
* reusable architecture design
* separation of model definition from preprocessing

---


#### `src/vision2d/processor.py`

This module handles preprocessing or task-specific logic for image-based inputs.

**It includes:**

* image loading
* resizing and normalization
* color space handling
* augmentation steps
* feature preparation for 2D models
* post-processing of predictions

**Concepts:**

* image preprocessing pipelines
* 2D feature extraction workflows
* preparing inputs for CNN-style or image-based models
