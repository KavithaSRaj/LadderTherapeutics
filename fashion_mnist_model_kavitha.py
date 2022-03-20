"""
Summary: Image Classification for Fashion MNIST data
Description: 
    Fashion MNIST dataset has 10 classes with image size of 28 * 28 each. A neaural network containing Convolution layers, Fully connected layers are implemented to predict the class of an image.
Dataset: Fashion MNIST
Training data size: 60000 ; Testing data size: 10000
@author: Kavitha Raj ; Date: Mar, 19, 2022

"""
# Install required prerequiste as specified in the Readme file before executing the code
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as transforms 
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import torcheck

# Open the configuration file which has the parameters set
with open('config.yaml') as cf_file:
    config = yaml.safe_load(cf_file.read())

# SummaryWriter to write into Tensorboard-specify the path
writer = SummaryWriter('runs/fashion_mnist')

# Get the Training data, decide the # of batches and load the train loader to run in specified # of batches on the training data
fashion_data = torchvision.datasets.FashionMNIST(root = '/',train=True, download=config['download'], transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]))
n_samples_train = fashion_data.__len__()
n_classes = len(fashion_data.targets.unique())
train_batch_size = int(n_samples_train/config['n_batches'])
train_loader = torch.utils.data.DataLoader(fashion_data, batch_size=train_batch_size, shuffle=True)
print("Total # of training data: ", n_samples_train)
print("# of batches for training data: ", config['n_batches'])
print("Size of each batch for training data: ", train_batch_size)

# Get the Testing data, decide the # of batches and load the test loader to run in specified # of batches on the training data
fashion_data = torchvision.datasets.FashionMNIST(root = '/',train=False, download=config['download'], transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]))
n_samples_test = fashion_data.__len__()
test_batch_size = int(n_samples_test/config['n_batches'])
test_loader = torch.utils.data.DataLoader(fashion_data, batch_size=test_batch_size, shuffle=True)
print("Total # of testing data: ", n_samples_test)
print("# of batches for testing data: ", config['n_batches'])
print("Size of each batch for testing data: ", test_batch_size)

# To display few sample images from the dataset

samples = iter(test_loader)
sample_images, sample_labels = samples.next()

# plot in the in the graph
for i in range(8):
    plt.subplot(4,4,i+1)
    plt.imshow(sample_images[i][0], cmap= 'gray')
plt.show()
image_grid = torchvision.utils.make_grid(sample_images)
# Display sample images in Tensorboard
writer.add_image('Fashion_mnist_sample_images', image_grid)

# Define the class, its attributes and methods for the classifier    
class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # first convolution 2d layer
        self.conv_1 = nn.Conv2d(in_channels=config['conv1_in_channels'],out_channels=config['conv1_out_channels'],kernel_size=config['conv1_kernel_size'])
        # Max pool layer
        self.pool = nn.MaxPool2d(kernel_size=config['pool_kernel_size'],stride=config['pool_stride'])
        # second convolution 2d layer
        self.conv_2 = nn.Conv2d(in_channels=config['conv2_in_channels'],out_channels=config['conv2_out_channels'],kernel_size=config['conv2_kernel_size'])
        # first fully connected layer
        self.fc_1 = nn.Linear(in_features=config['conv2_out_channels'] * config['conv2_kernel_size'] * config['conv2_kernel_size'], out_features=config['fc1_out_features'])
        # second fully connected layer
        self.fc_2 = nn.Linear(in_features=config['fc1_out_features'], out_features=config['fc2_out_features'])
        # final output fully connected layer
        self.fc_3 = nn.Linear(in_features=config['fc2_out_features'], out_features=config['fc3_out_features'])

    def forward(self, x):
        #reshape to set the input data in (#samples,#channels,Height,Width)
        x = torch.reshape(x,[x.shape[0],config['input_C'],config['input_H'],config['input_W']])
        # relu activation for first convolution layer and then pool
        x = self.pool(F.relu(self.conv_1(x)))
        # relu activation for second convolution layer and then pool
        x = self.pool(F.relu(self.conv_2(x)))
        # flatten to input to fully connected layer
        x = torch.flatten(x, 1)
        # relu activation for 1st fully connected layer
        x = F.relu(self.fc_1(x))
        # relu activation for 1st fully connected layer
        x = F.relu(self.fc_2(x))
        # final output layer
        x = self.fc_3(x)
        #return the output of the neural network
        return x

# Create an objetc of the classifier class
fmodel = FashionClassifier()

#Define the Criterion or loss function
loss_fn = nn.CrossEntropyLoss()

# Define the Optimizer along with the learning rate
optimizer = optim.Adam(fmodel.parameters(), lr = config['learning_rate'])

# Register the optimizer with Torcheck for verification
torcheck.register(optimizer)

# This is to check if the model parameters/weights are changing during the training
torcheck.add_module_changing_check(fmodel)

#To store training loss at each epoch
train_loss_vals =  []

# Write the graph of teh model architecture to Tensorboard
writer.add_graph(fmodel, sample_images.reshape(-1,28*28))

n_steps = len(train_loader)
training_correct = 0
# Train the model in a loop for specified number of epochs 
for epoch in range(config['num_epochs']):  # loop over the dataset multiple times
    training_loss = 0.0
    training_correct = 0
    train_epoch_loss= []
    # Get the training data in batches and train the model
    for X_train, y_train in train_loader:
        #clear the previously calculated gradients
        optimizer.zero_grad()
        #train the model with input images
        y_pred = fmodel(X_train.float())
        #check the loss incurred with predicted and actual labels
        loss = loss_fn(y_pred, y_train)
        #Backpropagation
        loss.backward()
        #Optimize
        optimizer.step()
        #calculate the running loss
        training_loss += loss.item()
        train_epoch_loss.append(loss.item())
        _, y_predt = torch.max(y_pred.data,1)
        #calculate the training accuracy
        training_correct += (y_predt == y_train).sum().item()
    # Write the training loss and accuracy results to Tensorboard
    writer.add_scalar('Training Loss', sum(train_epoch_loss)/len(train_epoch_loss), epoch+1)
    writer.add_scalar('Training Accuracy', 100*training_correct/n_samples_train, epoch+1)
    train_loss_vals.append(sum(train_epoch_loss)/len(train_epoch_loss))
    print(f"Epoch: {epoch+1}, Training loss: {(training_loss/len(train_loader)): .3f}, Training Accuracy: {(training_correct/n_samples_train): .2%}")



print('Training is completed')
# save the model for future use
#torch.save(fmodel.state_dict(), '/model.pth')

#Plot the training loss over each epoch
plt.title('Training Loss over epochs')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.plot(np.linspace(1, config['num_epochs'], config['num_epochs']).astype(int), train_loss_vals)
plt.show()
# display the summary of the model 
print(summary(fmodel, input_size = [1,28,28]))

#load the model for testing
#state_dict = torch.load('/model.pth')
#fmodel.load_state_dict(state_dict)
test_loss_vals = []

# Define the function to evaluate the model using the testing data
def model_eval():
    # Test the model
    fmodel.eval()
    testing_loss = 0.0
    accuracy = 0.0
    test_batch_loss = []
    labels = []
    predictions = []
    confusion_matrix = torch.zeros((n_classes, n_classes))
    # Turn of teh gradient calculation
    with torch.no_grad():
        n_samples = 0
        n_correct = 0
        i = 0
        # Load the testing data in batches and do teh predictions
        for X_test, y_test in test_loader:
            # pass the testing images to the trained model
            y_out = fmodel(X_test.float())
            #claculate the tetsing loss using the loss function
            loss = loss_fn(y_out, y_test)
            # Get the predictions, pick the classes with highest probabilities
            _, y_pred = torch.max(y_out.data,1)
            n_samples += y_test.size(0)
            #Calculate the accuracy
            n_correct +=(y_pred == y_test).sum().item()
            testing_loss += loss.item()
            test_batch_loss.append(loss.item())
            # Create confusion Matrix to understand the model performance better
            class_pred = [F.softmax(y, dim=0) for y in y_out]
            predictions.append(class_pred)
            labels.append(y_test)
            for t, p in zip(y_test.view(-1), y_pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            #write Model Accuracy and tetsing loss to Tensorbaord
            writer.add_scalar('Testing Loss', sum(test_batch_loss)/len(test_batch_loss), i+1)
            writer.add_scalar('Testing Accuracy', n_correct/test_batch_size, i+1)
            i = i + 1
        accuracy = 100.0* n_correct /n_samples
        # Write the pr curve, precision recall curve for each class to tensorboard
        labels = torch.cat(labels)
        predictions = torch.cat([torch.stack(pred) for pred in predictions])
        for i in range(n_classes):
            labels_i = labels == i
            predictions_i = predictions[:, i]
            writer.add_pr_curve(str(i), labels_i, predictions_i)
            writer.close()
        print(confusion_matrix)
        print(confusion_matrix.diag()/confusion_matrix.sum(1))
        print(f"Testing loss: {testing_loss/len(test_loader): .3f}, Test Accuracy: {accuracy/n_samples_test: .2%}")
    return test_batch_loss

#Call the model_eval function to test the trained model which returns the test loss
test_batch_loss = model_eval()

#plot training and testing loss in a graph
plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(test_batch_loss,label="test")
plt.plot(train_loss_vals,label="train")
plt.xlabel("Batches")
plt.ylabel("Loss")
plt.legend()
plt.show()
#Close the writer for Tensorboard
writer.close()

