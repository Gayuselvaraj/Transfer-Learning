# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
1. Develop a binary classification model using a pretrained VGG19 to distinguish between defected and non-defected capacitors by modifying the last layer to a single neuron.  
2. Train the model on a dataset containing images of various defected and non-defected capacitors to improve defect detection accuracy.  
3. Optimize and evaluate the model to ensure reliable classification for capacitor quality assessment in manufacturing.


## DESIGN STEPS
### STEP 1:
Collect and preprocess the dataset containing images of defected and non-defected capacitors.

### STEP 2:
Split the dataset into training, validation, and test sets.

### STEP 3:
Load the pretrained VGG19 model with weights from ImageNet.

### STEP 4:
Remove the original fully connected (FC) layers and replace the last layer with a single neuron (1 output) with a Sigmoid activation function for binary classification.

### STEP 5:
Train the model using binary cross-entropy loss function and Adam optimizer.

### STEP 6:
Evaluate the model with test data loader and intepret the evaluation metrics such as confusion matrix and classification report.

## PROGRAM

```python
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)

for param in model.parameters():
  param.requires_grad = False


```
```python
# Modify the final fully connected layer to match the dataset classes
num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features,1)

```
```python
# Include the Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)



```
```python
# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
```
```python
# Compute validation loss
model.eval()
val_loss = 0.0
with torch.no_grad():
  for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      outputs = torch.sigmoid(outputs)
      labels = labels.float().unsqueeze(1)
      loss = criterion(outputs, labels)
      val_loss += loss.item()
val_losses.append(val_loss / len(test_loader))
model.train()
print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

```
```python
# Plot training and validation loss
    print("Name: GAYATHRI S")
    print("Register Number: 212224230073")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="829" height="818" alt="Screenshot 2026-02-23 085912" src="https://github.com/user-attachments/assets/2fe2084b-034e-4cde-9cde-5b24446e2b2e" />


### Confusion Matrix

<img width="1351" height="659" alt="Screenshot 2026-02-23 072125" src="https://github.com/user-attachments/assets/7e49958a-c882-42d9-b077-504ac4663a70" />

### Classification Report
<img width="499" height="215" alt="image" src="https://github.com/user-attachments/assets/2ff78830-acd6-42fe-979d-7a925920d667" />

### New Sample Prediction
<img width="374" height="315" alt="Screenshot 2026-02-23 072133" src="https://github.com/user-attachments/assets/2d23c80e-6da9-479c-8bc0-749b19450057" />

<img width="322" height="326" alt="Screenshot 2026-02-23 072139" src="https://github.com/user-attachments/assets/18b59143-0584-47c1-bd77-dd44e2edfbdf" />


## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors
