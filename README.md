# Binary Polyp Classification with CNN

This file explains the implementation of Polyp-AI. Results can be found under RESULTS.md, and the main python file can be found under polyp_ai.py. Finally, any credits/sources are found under CREDITS.md .

This project involves training a Convolutional Neural Network (CNN) for binary classification of polyp images. The model is trained to classify images into two categories: `negativeOnly` and `positive`.

## Project Structure

- **Data Directory**: `sequenceData`
  - `negativeOnly`: Directory containing images without polyps.
  - `positive`: Directory containing images with polyps.
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

### Installation

Install the required packages using pip:

```bash
pip install torch torchvision matplotlib numpy tqdm pillow

## Training the Model

The model can be trained using the provided hyperparameters. The training and testing steps include the calculation of loss and accuracy for each epoch.

### Training and Evaluation Loop

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
