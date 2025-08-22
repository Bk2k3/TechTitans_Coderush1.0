
# TechTitans_Coderush1.0

EEG-based schizophrenia detection powered by machine learning.

---

##  Table of Contents

- [Project Overview](#project-overview)  
- [Setup & Installation](#setup--installation)  
  - [1. Clone the Repository](#1-clone-the-repository)  
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)  
  - [3. Install Dependencies](#3-install-dependencies)  
- [Model Options](#model-options)  
  - [Option A: Download Pre-trained Model](#option-a-download-pre-trained-model)  
  - [Option B: Train Your Own Model](#option-b-train-your-own-model)  
- [Running the App](#running-the-app)  
- [File Structure](#file-structure)  
- [Results](#results)  
- [Additional Notes](#additional-notes)  
- [License](#license)

---

## Project Overview

This project implements an EEG-based schizophrenia detection solution using machine learning.  
It includes both a trained model (`eeg_schizophrenia_model.pt`) and a training script (`training_script.py`), with a Streamlit interface (`run.py`) for ease of use.

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Bk2k3/TechTitans_Coderush1.0.git
cd TechTitans_Coderush1.0


### 2. Create a Virtual Environment

Set up an isolated Python environment to manage dependencies cleanly:

```bash
# Using venv (Python 3.x)
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Or using conda
conda create -n codrush1 python=3.10
conda activate codrush1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Options

You can choose between using the existing pre-trained model or training your own from scratch:

### Option A: Download Pre-trained Model

The pre-trained model files (e.g., `eeg_schizophrenia_model.pt`, `eeg_scaler.pkl`, and `model_metadata.json`) are already provided in the repo.
Ensure they remain in the root folder so the app can load them when running.

### Option B: Train Your Own Model

1. Prepare your EEG dataset in the expected format.

2. Run the training script:

   ```bash
   python training_script.py
   ```

3. Upon completion, the script will output:

   * A new model file (e.g., `eeg_schizophrenia_model.pt`)
   * A scaler (`eeg_scaler.pkl`)
   * Metadata (`model_metadata.json`)
   * And a results image (`training_results.png`)

4. Replace the existing model-related files in the repository root with your new outputs.

---

## Running the App

Once your virtual environment is activated and dependencies are installed:

```bash
streamlit run run.py
```

This will open the Streamlit interface in your default browser.
Use it to upload EEG data or explore the model's predictions interactively.

---

## File Structure

```
TechTitans_Coderush1.0/
├── run.py                     # Streamlit app entry point
├── training_script.py         # Script to train your own model
├── eeg_schizophrenia_model.pt # Pre-trained PyTorch model
├── eeg_scaler.pkl             # Pre-processing scaler
├── model_metadata.json        # Metadata for model & scaler
├── training_results.png       # Illustration of training results
├── .gitignore
└── requirements.txt           # Project dependencies
```

---

## Results

The following image shows training performance and results:

![Training Results](training_results.png)

The graph provides insights into model accuracy, loss, and convergence during training.

---

## Additional Notes

* If you download the project as a ZIP (instead of cloning), you’ll need to initialize Git manually and push to your own repo if needed:

  ```bash
  git init
  git remote add origin <your-repo-url>
  git add .
  git commit -m "Initial commit"
  git push -u origin main
  ```

* For any issues related to model loading or dependencies, check version compatibility (e.g., PyTorch, Streamlit).

---


## Contact / Support

For questions or help, open an issue in the repo or reach out to the project maintainer.

