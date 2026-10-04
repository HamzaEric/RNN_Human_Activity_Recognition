# RNN Human Activity Recognition

This repository contains a PyTorch implementation of a Recurrent Neural Network (RNN) designed for Many-to-One sequence classification. The model is trained on the UCI Human Activity Recognition (HAR) dataset to classify 6 distinct physical activities based on 9-channel smartphone sensor data (accelerometer and gyroscope) over 128-timestep windows.

## Repository Structure

The repository is organized into a main directory containing this `README.md` file, alongside two primary folders: `Model` and `Notebooks`[cite: 4].

### Notebooks
The `Notebooks` directory contains the core exploratory and training code, broken down into four distinct phases[cite: 5]:

*   **`Vanilla_RNN.ipynb`**: Covers data loading, preprocessing (per-channel Z-score standardization), and the implementation of the baseline stacked vanilla RNN (e.g., `num_layers=2`, `hidden_size=8`, `nonlinearity='relu'`). Includes training loops and loss plotting.
*   **`RNNs.ipynb`**: Explores alternate DataLoader configurations, varying batch sizes, and model iterations.
*   **`Hidden_states_visualization.ipynb`**: Provides diagnostic tools to explore internal network memory. Includes functions to isolate high-variance hidden dimensions and plot 2D hidden state trajectories using PCA (e.g., analyzing the "LAYING" vs. "SITTING" states).
*   **`Vanishing_Gradients.ipynb`**: Analyzes the step-to-step hidden change ($\vert{}\vert{}h_t - h_{t-1}\vert{}\vert{}$) and computes per-timestep input gradient norms ($\vert{}\vert{}dL/dx_t\vert{}\vert{}$) to diagnose early-timestep gradient vanishing.

### Model
The `Model` directory is used for storing trained weights and network artifacts[cite: 6]. 
*   **`rnn_har.pth`**: The saved `state_dict` for the evaluated RNN model (configured with `hidden_size=32`).

## Dataset
This project relies on the **UCI HAR Dataset**. The data pipeline ingests standardized windows of shape `(Batch, 128, 9)` representing:
*   3 channels of Total Acceleration (X, Y, Z)
*   3 channels of Body Acceleration (X, Y, Z)
*   3 channels of Body Gyroscope (X, Y, Z)

The network maps the final hidden state ($h_{128}$) to one of 6 activity classes: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, and `LAYING`.

## Setup & Execution

1. Clone the repository.
2. Ensure you have `torch`, `numpy`, `pandas`, `matplotlib`, and `scikit-learn` installed.
3. Open the `Notebooks/` directory[cite: 5] in Jupyter or Google Colab and run `Vanilla_RNN.ipynb`[cite: 5] to establish the baseline standardization stats and model.
4. To evaluate the pre-trained model without retraining, load `Model/rnn_har.pth`[cite: 6] into an `RNNClassifier` instance with a matching `hidden_size`.
