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
#### Model Summary
cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 28, 28]             320
            Conv2d-2           [-1, 64, 28, 28]          18,496
         MaxPool2d-3           [-1, 64, 14, 14]               0
            Conv2d-4          [-1, 128, 14, 14]          73,856
            Conv2d-5          [-1, 256, 14, 14]         295,168
         MaxPool2d-6            [-1, 256, 7, 7]               0
            Conv2d-7            [-1, 512, 5, 5]       1,180,160
            Conv2d-8           [-1, 1024, 3, 3]       4,719,616
            Conv2d-9             [-1, 10, 1, 1]          92,170
================================================================
Total params: 6,379,786
Trainable params: 6,379,786
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.51
Params size (MB): 24.34
Estimated Total Size (MB): 25.85

#### Train and Test Accuracy obtained for diffetent epochs
EPOCH: 0
  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
Loss=0.171647310256958 Batch_id=468 Accuracy=86.91: 100%|██████████| 469/469 [00:37<00:00, 12.53it/s]

Test set: Average loss: 0.0709, Accuracy: 9778/10000 (97.78%)

EPOCH: 1
Loss=0.02345419116318226 Batch_id=468 Accuracy=98.33: 100%|██████████| 469/469 [00:37<00:00, 12.51it/s]

Test set: Average loss: 0.0439, Accuracy: 9849/10000 (98.49%)

EPOCH: 2
Loss=0.05811501666903496 Batch_id=468 Accuracy=98.95: 100%|██████████| 469/469 [00:37<00:00, 12.49it/s]

Test set: Average loss: 0.0380, Accuracy: 9872/10000 (98.72%)

EPOCH: 3
Loss=0.008012492209672928 Batch_id=468 Accuracy=99.20: 100%|██████████| 469/469 [00:37<00:00, 12.51it/s]

Test set: Average loss: 0.0305, Accuracy: 9897/10000 (98.97%)

EPOCH: 4
Loss=0.09363314509391785 Batch_id=468 Accuracy=99.48: 100%|██████████| 469/469 [00:37<00:00, 12.54it/s]

Test set: Average loss: 0.0262, Accuracy: 9916/10000 (99.16%)

EPOCH: 5
Loss=0.004521720577031374 Batch_id=468 Accuracy=99.54: 100%|██████████| 469/469 [00:37<00:00, 12.55it/s]

Test set: Average loss: 0.0299, Accuracy: 9907/10000 (99.07%)

EPOCH: 6
Loss=0.01575702615082264 Batch_id=468 Accuracy=99.62: 100%|██████████| 469/469 [00:37<00:00, 12.52it/s]

Test set: Average loss: 0.0284, Accuracy: 9918/10000 (99.18%)

EPOCH: 7
Loss=0.0011563700390979648 Batch_id=468 Accuracy=99.75: 100%|██████████| 469/469 [00:37<00:00, 12.54it/s]

Test set: Average loss: 0.0268, Accuracy: 9910/10000 (99.10%)

EPOCH: 8
Loss=8.187795901903883e-05 Batch_id=468 Accuracy=99.83: 100%|██████████| 469/469 [00:37<00:00, 12.56it/s]

Test set: Average loss: 0.0304, Accuracy: 9914/10000 (99.14%)

EPOCH: 9
Loss=0.0015095145208761096 Batch_id=468 Accuracy=99.82: 100%|██████████| 469/469 [00:37<00:00, 12.56it/s]

Test set: Average loss: 0.0253, Accuracy: 9934/10000 (99.34%)

EPOCH: 10
Loss=0.0019395098788663745 Batch_id=468 Accuracy=99.84: 100%|██████████| 469/469 [00:37<00:00, 12.52it/s]

Test set: Average loss: 0.0293, Accuracy: 9912/10000 (99.12%)

EPOCH: 11
Loss=0.0034592943266034126 Batch_id=468 Accuracy=99.84: 100%|██████████| 469/469 [00:37<00:00, 12.49it/s]

Test set: Average loss: 0.0286, Accuracy: 9922/10000 (99.22%)

EPOCH: 12
Loss=0.0008882636320777237 Batch_id=468 Accuracy=99.87: 100%|██████████| 469/469 [00:37<00:00, 12.53it/s]

Test set: Average loss: 0.0303, Accuracy: 9922/10000 (99.22%)

EPOCH: 13
Loss=0.0002032176562352106 Batch_id=468 Accuracy=99.93: 100%|██████████| 469/469 [00:37<00:00, 12.53it/s]

Test set: Average loss: 0.0307, Accuracy: 9923/10000 (99.23%)

EPOCH: 14
Loss=0.01875772513449192 Batch_id=468 Accuracy=99.93: 100%|██████████| 469/469 [00:37<00:00, 12.49it/s]

Test set: Average loss: 0.0309, Accuracy: 9925/10000 (99.25%)

EPOCH: 15
Loss=0.0022081320639699697 Batch_id=468 Accuracy=99.93: 100%|██████████| 469/469 [00:37<00:00, 12.50it/s]

Test set: Average loss: 0.0312, Accuracy: 9923/10000 (99.23%)

EPOCH: 16
Loss=1.0702807230700273e-05 Batch_id=468 Accuracy=99.93: 100%|██████████| 469/469 [00:37<00:00, 12.49it/s]

Test set: Average loss: 0.0321, Accuracy: 9930/10000 (99.30%)

EPOCH: 17
Loss=4.736737537314184e-05 Batch_id=468 Accuracy=99.98: 100%|██████████| 469/469 [00:37<00:00, 12.52it/s]

Test set: Average loss: 0.0303, Accuracy: 9932/10000 (99.32%)

EPOCH: 18
Loss=3.0197588785085827e-05 Batch_id=468 Accuracy=100.00: 100%|██████████| 469/469 [00:37<00:00, 12.53it/s]

Test set: Average loss: 0.0329, Accuracy: 9932/10000 (99.32%)

EPOCH: 19
Loss=0.00012379475811030716 Batch_id=468 Accuracy=99.99: 100%|██████████| 469/469 [00:37<00:00, 12.52it/s]

Test set: Average loss: 0.0347, Accuracy: 9930/10000 (99.30%)

####Resuts
- parameters : >6M
- Best Training Accuracy: 99.99
- Best Test Accuracy: 99.30

####Analysis
- we see the number of parameters is too high for our model. 
- As train Accuracy is more than test accuracy for a significant number of epochs we can observe that our model is overfitting. 

Let us build another network to improve our results.

### Step 2.
#### Target
- Reduce the number of parameters to less than 10K
- Reduce the overfitting model by adding dropout and increase efficiency by adding batchnorm.
#### Defined network
``` python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
             nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
             nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.dropout(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.dropout(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
```
