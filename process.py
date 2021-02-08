import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Getting the input path of the file up to the directory
input_path = os.path.dirname(os.path.realpath(__file__))


# Extracting the training and testing datasets
df_train = pd.read_csv(input_path + '/train.csv')
df_test = pd.read_csv(input_path + '/test.csv')

# Splitting the testing and training data
split = np.random.rand(len(df_train)) < .8
df_val = df_train[~split]
df_train = df_train[split]

# Defining the dataset. Inheriting from the torch.data.utils.Dataset class that returns len(dataset) as a length and the allows indexing through the dunder methods below
class MNISTDataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, n):
        data = self.df.iloc[n]
        image = data[1:].values.reshape((28, 28)).astype(np.uint8)
        label = data[0]
        if self.transform:
            image = self.transform(image)
        return (image, label)

    def tonumpy(self):
        data

# Defining the data augmentation/transforms (and the batch size for the data loaders below)
batch_size = 16
classes = range(10)

# Defining the mean and standard deviation for the normalization of the values in the image pixels
train_data = df_train.drop('label', axis=1).values
train_mean = train_data.mean()/255
train_std = train_data.std()/255

# Defining the transform
train_transform = transforms.Compose([

    # Converting the array into a PIL image
    transforms.ToPILImage(),

    # Random affination for the image. Tilt up to 20 degrees, translate x and y by up to10 percent, scale up or down by 10 percent
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),

    # Color Jittering for random brightness changes and contrast changes
    transforms.ColorJitter(brightness=0.2, contrast=0.2),

    # Converting the image into a tensor for the matrix manipulation
    transforms.ToTensor(),

    # Normalizing the pixel values to avoid ICS from the varying pixel value distributions
    transforms.Normalize(mean=[train_mean], std=[train_std]),

    ])

val_transform = transforms.Compose([

    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_mean], std=[train_std]),

    ])

# Setting the testing transform to the validation transform (both are the same)
test_transform = val_transform

# Transforming the dataframes into data loaders
train_dataset = MNISTDataset(df_train, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = MNISTDataset(df_val, transform=val_transform)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Defining the neural network
class MNISTModel(nn.Module):

    def __init__(self):
        super(MNISTModel, self).__init__()

        # Defining the first section of the neural network
        self.conv1 = nn.Sequential(

                # Starting with a 2d conv layer that takes in the single channel then produces 32
                nn.Conv2d(1, 32, 3, padding=1),

                # Activation for the nodes
                nn.ReLU(),

                # Batch normalization for the 16 images and their data. We take in all 32 channels to normalize
                nn.BatchNorm2d(32),

                # Creating another layer of 32 channels output
                nn.Conv2d(32, 32, 3, padding=1),

                # Activationg for the nodes
                nn.ReLU(),

                # Normalizing all 32 channels
                nn.BatchNorm2d(32),

                # Another layer of convolution, this time with a different stride
                nn.Conv2d(32, 32, 3, stride=2, padding=1),

                # Activation once again
                nn.ReLU(),

                # Batch Normalization
                nn.BatchNorm2d(32),

                # Pooling the pixels by their maximum value. A kernel of size 2x2 and a stride of 2.
                nn.MaxPool2d(2, 2),

                # Defining the dropout that will be applied the entire section
                nn.Dropout(.25),

            )

        # Not going to comment on all of these since they are just variations of the same code commented above
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25)
            )

        self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25)
            )
        
        # This is where all the channels for each image is going to get blobbed into one single row, and then classified for which number it could be
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 10)
        )

    def forward(self, x):

        # Passing the data through the layers we have defined up above
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flattening the tensor to be only of 2d (or just flattening to the first dimenstion)
        x = self.fc(x)

        # Returning the log to account for any super small values and still have accuracy for loss and accuracy itself
        # The softmax of all the images in the batch, so we look at the first dimention(the features of the image). The 0th dimension is the whole batch, which if we tried to softmax it,
        # would weirdly give a probability distribution for the position of the image in the batch rather than the features of the image.
        x = F.log_softmax(x, dim=1)

        return x


# Instantiating the model, optimizer, loss function, and sending the model onto our device (the cpu)
model = MNISTModel()
model.to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# Defining the training process
epochs = 3
train_losses, val_losses = [], []
train_accu, val_accu = [], []
start_time = time.time()
early_stop_counter = 10
counter = 0
best_val_loss = float('Inf')

for epoch in range(epochs):
    epoch_start_time = time.time()
    running_loss = 0
    accuracy = 0

    # Training for one epoch
    model.train()
    
    # Printing the epoch
    print("Epoch: {}/{}.. ".format(epoch+1, epochs)

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Zeroing optimizer buffer
        optimizer.zero_grad()

        # Getting the logarithmic prediction set
        log_ps = model(images)

        # Taking the natural log off to see the probabilities
        ps = torch.exp(log_ps)

        # Returning the largest probability and index of class for each of the 16 images in the batch(the classification with the largest probability found in the softmax function)
        # topk returns both the top k values and the indexes of those values. We are returning the most probable number guess for each image in the batch, along with the index (or in this case the classification)
        # for each image in the batch as well. So in total, 16 highest probabilities and their 16 classifications to go with it.
        top_p, top_class = ps.topk(1, dim=1)

        # Viewing the label as the same shape as the top class (16 guessed images) and getting the number of correctly guessed images
        # This "equals" tensor is full of 0s and 1s for true and false
        equals = top_class == labels.view(*top_class.shape)

        # The average of the guesses so 1 is perfect and 0 is completely incorrect. Sort of like a grade ---> accuracy.
        # This process accumulates the accuracy of each batch of images and will go through the entire training dataset.
        accuracy += torch.mean(equals.type(torch.FloatTensor))

        # Using NLLoss as mathematically recommended for the criterion the batch sizes and guesses are matched thanks to the data loader and the loss function with the neural network
        # working together. I think it utilizes the last node of the model to identify the classifications.
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        # Adding the running loss to calculate the average loss per training epoch below
        # This script will accumulate the loss of each batch of images and will go through the entire training dataset.
        running_loss += loss.item()


    # Record training loss and error per epoch (which is all the training data one time), then evaluate using validation data
    train_losses.append(running_loss/len(train_loader))
    train_accu.append(accuracy/len(train_loader))
    val_loss = 0
    accuracy=0

    # Evaluating the model. Turning off the batch normalization, dropout, and then deactivating autograd to speed up the process/use less compute.
    model.eval()

    with torch.no_grad():

        # Same process as training, except no learning
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Getting the output for the batch
            log_ps = model(images)
            
            # Adding to the validation loss (one epoch at a time for graphing)
            val_loss += criterion(log_ps, labels)

            # Getting the prediction set with softmax probability
            ps = torch.exp(log_ps)

            # Getting the top probability for each image in the batch along with their corresponding classification (or index)
            top_p, top_class = ps.topk(1, dim=1)

            # Getting the correctly classified predictions. We use the star to look at each classification in the batch.
            equals = top_class == labels.view(*top_class.shape)

            # Calculating the average accuracy for this batch's correctness
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    # Getting the validation loss to add for this epoch
    val_losses.append(val_loss/len(val_loader))
    val_accu.append(accuracy/len(val_loader))

    # Printing out the progress (might use tqdm up above for this instead)
    print("Time: {:.2f}s..".format(time.time()-epoch_start_time),
         "Training Loss: {:.3f}.. ".format(train_losses[-1]),
         "Training Accu: {:.3f}.. ".format(train_accu[-1]),
         "Val Loss: {:.3f}.. ".format(val_losses[-1]),
         "Val Accu: {:.3f}".format(val_accu[-1]))

    # We check to make sure there has been improvement from at least the last ten epochs and then stop if we haven't
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        counter = 0
        best_model_wts = copy.deepcopy(model.state_dict())
    else:
        counter += 1
        print('Validation loss has not improved since: {:.3f}..'.format(best_val_loss), 'Count: ', str(counter))

        # Stopping early and then reverting the model into it's previous best state that was saved in the deepcopy above
        if counter >= early_stop_counter:
            print('Early Stopping Now!!!!')
            model.load_state_dict(best_model_wts)
            break

# plot training history
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
ax = plt.gca()
ax.set_xlim([0, epoch + 2])
plt.ylabel('Loss')
plt.plot(range(1, epoch + 2), train_losses[:epoch+1], 'r', label='Training Loss')
plt.plot(range(1, epoch + 2), val_losses[:epoch+1], 'b', label='Validation Loss')
ax.grid(linestyle='-.')
plt.legend()
plt.subplot(2,1,2)
ax = plt.gca()
ax.set_xlim([0, epoch+2])
plt.ylabel('Accuracy')
plt.plot(range(1, epoch + 2), train_accu[:epoch+1], 'r', label='Training Accuracy')
plt.plot(range(1, epoch + 2), val_accu[:epoch+1], 'b', label='Validation Accuracy')
ax.grid(linestyle='-.')
plt.legend()
plt.show()


# DataFrame.values will convert each row into a index and at each index the columns will become a list at its corresponding
x_test = df_test.values

# The df_test.values (x_test) is already a list of 28 by 28 lists(784 pixels big), but we need it to be in float mode to normalize to the accrate decimal value
x_test = x_test.reshape([-1, 28, 28]).astype(np.float)

# Getting the pixel values of the testing data to be between 0 and 1, doing standard normalization to get the concentration 
x_test = x_test/255.

# Z-Score Normalizing the testing data with the training mean and standard deviation
x_test = (x_test-train_mean)/train_std

# Resizing the numpy array to be a list of list of rows. Meaning x_test is now a list that contains lists of rows at each index of the list
x_test = np.expand_dims(x_test, axis=1)

# Then somehow converting the strangly shaped numpy array above into a float tensor for processing in the model
x_test = torch.from_numpy(x_test).float().to(device) 

# Making predictions
with torch.no_grad():

    # Passing in entire dataloader
    ps = model(test_loader)

    # Getting the largest probability for each of the predicted images
    prediction = torch.argmax(ps, 1)

# Preparing the output file
df_export = pd.DataFrame(prediction.cpu().tolist(), columns=['Label'])

# Adding the image id column to our dataframe-to-import
df_export['ImageId'] = df_export.index + 1

# Rearranging the export file to match the submission structure
df_export = df_export[['ImageId', 'Label']]
df_export.to_csv('output.csv', index=False)














