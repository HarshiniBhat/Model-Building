# Model-Building
 Aim -
  - To reach an accuracy of 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
	- In Less than or equal to 20 Epochs (is can do in 15 better)
	- With Less than 10000 Parameters
	- To Do this in exactly 5 steps
	
  ## Approach to the problem
  - First we shall design the skeleton of the network and apply different techniques to obtain the best model.
  The following steps are carried out initially to frame the skelton 
  - Importing the required libraries

- Downloading the data
- Defining a dataset class
- Using the defined class and getting the data into the train_loader, test_loader
- Visualising the data.
- Defining the network
- Training the model on the defined network
- plotting the loss and accuracy curve against different epochs for the test and train set.

## we shall see the results and analysis of the model in every step 
### Network 1. 
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #input -? OUtput? RF
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 1024, 3)
        self.conv7 = nn.Conv2d(1024, 10, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
 ```
