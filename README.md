# Binary Classification of Polyps (Cancer Precursors) through CNNs


This project involves training a Convolutional Neural Network (CNN) for binary classification of polyp images. Polyps are precursors for cancer found durign colonoscopies. The model is trained to classify images into two categories: `Positive` and `Negative`.

- Script to train model can be found under `polyp_ai.py`
- Trained model can be found under `Polyp_AI_Model_V0-2.pth`
- Performace / Results can be found under `RESULTS.md`
- Citations can be found under `CITATIONS.md`

## Disclaimer

- To train the model, there is no dataset on Github. Dataset is extracted from Synapse once you run the script locally
- Use your own API token on line 40 to extract the data from synapse client
- Lines 13 - 64 and 470 - end of file contain all code used to train the model
- The commented section between 64 - 470 of the script was my attempt at implementing an Object detection model with the use of CNNs that outputs masks.

## Project Structure

- **Model**: `BinaryClassificationCNN`
- **Hyperparameters**: 
  - Learning rate: `0.0005`
  - Number of workers: `2`
  - Input channels: `3`
  - Hidden units: `32`
  - Number of batches: `16`
  - Number of epochs: `3`
  - Image size: `128`
  - Device: `cuda` if available, else `cpu`

## Setup

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm
- Pillow
- synapseclient

### Installation

Install the required packages using pip/pip3:

```bash
pip3 install torch torchvision matplotlib numpy tqdm pillow synapseclient
```

### Training and Evaluation Loop

The model uses binary classification with logits loss as the loss function. The optimizer used was Adam. 

This code was device agnostic, meaning both your computer's CPU or a Google Colab GPU can be used to run the script.

```python
# Initialize the model, loss function, and optimizer
model = BinaryClassificationCNN(input_channels, hidden_units).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
train_and_evaluate(model, train_loader, test_loader, loss_fn, optimizer, accuracy_fn, device, epochs)
```

## Visualizing Predictions

The project includes functionality to visualize random predictions from the test set. The visualization displays 9 images in a 3x3 grid with true labels, predicted labels, and prediction probabilities.

### Visualization

```python
# Visualize predictions
visualize_predictions(model, test_loader, device, num_images=9)
```

## Confusion Matrix

To evaluate the model's performance, a confusion matrix can be generated to display True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

### Confusion Matrix

```python
# Generate confusion matrix
conf_matrix = generate_confusion_matrix(model, test_loader, device)
```

## Conclusion

This project demonstrates the use of a CNN for binary classification of polyp images. The model's performance is evaluated using accuracy, loss, and a confusion matrix to analyze the classification results.

Feel free to modify the hyperparameters and experiment with different configurations to improve the model's performance.

## License

This project is licensed under the MIT License.
