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
### Step 1
### Network  
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

        Layer (type)               Output Shape         Param #

            Conv2d-1           [-1, 32, 28, 28]             320
            Conv2d-2           [-1, 64, 28, 28]          18,496
         MaxPool2d-3           [-1, 64, 14, 14]               0
            Conv2d-4          [-1, 128, 14, 14]          73,856
            Conv2d-5          [-1, 256, 14, 14]         295,168
         MaxPool2d-6            [-1, 256, 7, 7]               0
            Conv2d-7            [-1, 512, 5, 5]       1,180,160
            Conv2d-8           [-1, 1024, 3, 3]       4,719,616
            Conv2d-9             [-1, 10, 1, 1]          92,170

Total params: 6,379,786
Trainable params: 6,379,786

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

#### Resuts
- parameters : >6M
- Best Training Accuracy: 99.99
- Best Test Accuracy: 99.30

#### Analysis
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
#### Train and Test Accuracy obtained for diffetent epochs

EPOCH: 0
  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
Loss=0.0516323447227478 Batch_id=468 Accuracy=93.14: 100%|██████████| 469/469 [00:17<00:00, 26.07it/s]

Test set: Average loss: 0.0614, Accuracy: 9811/10000 (98.11%)

EPOCH: 1
Loss=0.05035039782524109 Batch_id=468 Accuracy=97.98: 100%|██████████| 469/469 [00:18<00:00, 25.96it/s]

Test set: Average loss: 0.0599, Accuracy: 9810/10000 (98.10%)

EPOCH: 2
Loss=0.08698663115501404 Batch_id=468 Accuracy=98.32: 100%|██████████| 469/469 [00:17<00:00, 26.19it/s]

Test set: Average loss: 0.0382, Accuracy: 9880/10000 (98.80%)

EPOCH: 3
Loss=0.018007507547736168 Batch_id=468 Accuracy=98.56: 100%|██████████| 469/469 [00:17<00:00, 26.13it/s]

Test set: Average loss: 0.0403, Accuracy: 9873/10000 (98.73%)

EPOCH: 4
Loss=0.058485787361860275 Batch_id=468 Accuracy=98.64: 100%|██████████| 469/469 [00:17<00:00, 26.59it/s]

Test set: Average loss: 0.0433, Accuracy: 9857/10000 (98.57%)

EPOCH: 5
Loss=0.048019036650657654 Batch_id=468 Accuracy=98.76: 100%|██████████| 469/469 [00:17<00:00, 26.50it/s]

Test set: Average loss: 0.0319, Accuracy: 9888/10000 (98.88%)

EPOCH: 6
Loss=0.039690565317869186 Batch_id=468 Accuracy=98.93: 100%|██████████| 469/469 [00:17<00:00, 26.53it/s]

Test set: Average loss: 0.0338, Accuracy: 9890/10000 (98.90%)

EPOCH: 7
Loss=0.007865828461945057 Batch_id=468 Accuracy=98.95: 100%|██████████| 469/469 [00:17<00:00, 26.37it/s]

Test set: Average loss: 0.0327, Accuracy: 9898/10000 (98.98%)

EPOCH: 8
Loss=0.03466476872563362 Batch_id=468 Accuracy=98.97: 100%|██████████| 469/469 [00:17<00:00, 26.09it/s]

Test set: Average loss: 0.0275, Accuracy: 9913/10000 (99.13%)

EPOCH: 9
Loss=0.022824542596936226 Batch_id=468 Accuracy=99.08: 100%|██████████| 469/469 [00:18<00:00, 26.02it/s]

Test set: Average loss: 0.0301, Accuracy: 9905/10000 (99.05%)

EPOCH: 10
Loss=0.024445118382573128 Batch_id=468 Accuracy=99.14: 100%|██████████| 469/469 [00:17<00:00, 26.15it/s]

Test set: Average loss: 0.0286, Accuracy: 9908/10000 (99.08%)

EPOCH: 11
Loss=0.014252151362597942 Batch_id=468 Accuracy=99.10: 100%|██████████| 469/469 [00:18<00:00, 26.03it/s]

Test set: Average loss: 0.0300, Accuracy: 9905/10000 (99.05%)

EPOCH: 12
Loss=0.008483725599944592 Batch_id=468 Accuracy=99.21: 100%|██████████| 469/469 [00:18<00:00, 25.81it/s]

Test set: Average loss: 0.0270, Accuracy: 9917/10000 (99.17%)

EPOCH: 13
Loss=0.03999733552336693 Batch_id=468 Accuracy=99.23: 100%|██████████| 469/469 [00:18<00:00, 25.83it/s]

Test set: Average loss: 0.0262, Accuracy: 9912/10000 (99.12%)

EPOCH: 14
Loss=0.0381893664598465 Batch_id=468 Accuracy=99.28: 100%|██████████| 469/469 [00:18<00:00, 25.98it/s]

Test set: Average loss: 0.0266, Accuracy: 9920/10000 (99.20%)

#### Resuts
- Parameters : 9,950
- Best Training Accuracy: 99.28
- Best Test Accuracy: 99.20 
#### Analysis
- The model has now lesser number of parameters 
-Initially the model was underfitting then it started overfitting hence we can optimise our network to have a slightly underfitting network as we cannot train harder over this model.

## Step 3.
#### Target:
  Adding GAP to get a better model and reduce the overfitting

#### Defined Network
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
        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
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
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

#### Train and Test Accuracy obtained for diffetent epochs

EPOCH: 0
  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
Loss=0.3310699164867401 Batch_id=468 Accuracy=79.87: 100%|██████████| 469/469 [00:21<00:00, 22.09it/s]

Test set: Average loss: 0.5331, Accuracy: 8561/10000 (85.61%)

EPOCH: 1
Loss=0.14782045781612396 Batch_id=468 Accuracy=95.43: 100%|██████████| 469/469 [00:21<00:00, 22.27it/s]

Test set: Average loss: 0.1840, Accuracy: 9639/10000 (96.39%)

EPOCH: 2
Loss=0.1896640807390213 Batch_id=468 Accuracy=96.61: 100%|██████████| 469/469 [00:20<00:00, 22.46it/s]

Test set: Average loss: 0.1371, Accuracy: 9733/10000 (97.33%)

EPOCH: 3
Loss=0.08711912482976913 Batch_id=468 Accuracy=97.07: 100%|██████████| 469/469 [00:20<00:00, 22.46it/s]

Test set: Average loss: 0.1406, Accuracy: 9645/10000 (96.45%)

EPOCH: 4
Loss=0.08043765276670456 Batch_id=468 Accuracy=97.39: 100%|██████████| 469/469 [00:20<00:00, 22.54it/s]

Test set: Average loss: 0.1016, Accuracy: 9756/10000 (97.56%)

EPOCH: 5
Loss=0.16175001859664917 Batch_id=468 Accuracy=97.63: 100%|██████████| 469/469 [00:21<00:00, 22.15it/s]

Test set: Average loss: 0.0904, Accuracy: 9790/10000 (97.90%)

EPOCH: 6
Loss=0.10000065714120865 Batch_id=468 Accuracy=97.78: 100%|██████████| 469/469 [00:21<00:00, 22.00it/s]

Test set: Average loss: 0.1059, Accuracy: 9711/10000 (97.11%)

EPOCH: 7
Loss=0.07069698721170425 Batch_id=468 Accuracy=97.89: 100%|██████████| 469/469 [00:21<00:00, 22.27it/s]

Test set: Average loss: 0.0854, Accuracy: 9762/10000 (97.62%)

EPOCH: 8
Loss=0.07804854959249496 Batch_id=468 Accuracy=97.97: 100%|██████████| 469/469 [00:21<00:00, 22.12it/s]

Test set: Average loss: 0.0737, Accuracy: 9795/10000 (97.95%)

EPOCH: 9
Loss=0.08454219251871109 Batch_id=468 Accuracy=97.98: 100%|██████████| 469/469 [00:21<00:00, 22.24it/s]

Test set: Average loss: 0.0661, Accuracy: 9818/10000 (98.18%)

EPOCH: 10
Loss=0.04058723524212837 Batch_id=468 Accuracy=98.13: 100%|██████████| 469/469 [00:21<00:00, 22.25it/s]

Test set: Average loss: 0.0709, Accuracy: 9817/10000 (98.17%)

EPOCH: 11
Loss=0.1343642771244049 Batch_id=468 Accuracy=98.15: 100%|██████████| 469/469 [00:21<00:00, 22.27it/s]

Test set: Average loss: 0.0854, Accuracy: 9755/10000 (97.55%)

EPOCH: 12
Loss=0.060611456632614136 Batch_id=468 Accuracy=98.13: 100%|██████████| 469/469 [00:21<00:00, 22.12it/s]

Test set: Average loss: 0.0783, Accuracy: 9789/10000 (97.89%)

EPOCH: 13
Loss=0.06750783324241638 Batch_id=468 Accuracy=98.30: 100%|██████████| 469/469 [00:21<00:00, 22.23it/s]

Test set: Average loss: 0.0756, Accuracy: 9784/10000 (97.84%)

EPOCH: 14
Loss=0.06243039295077324 Batch_id=468 Accuracy=98.32: 100%|██████████| 469/469 [00:21<00:00, 22.07it/s]

Test set: Average loss: 0.0621, Accuracy: 9833/10000 (98.33%)


#### Resuts:
- Parameters : 5050
- Best Training Accuracy: 97.84%
- Best Test Accuracy:98.33%


#### Analysis:
The model accuracy is not that good enough and it is a slightly underfitting model and the accuracy has dropped slightly compared to the previous step.
The drop in the accuracy is not because we have added GAP. As comparitively with such a light model we have reached accuracy of 98.33%. which is a good number for such a light model.

## Step 4.
#### Target:
Since the accuracy has dropped and  as  model capacity has reduced after adding GAP, we will increase the model capacity by adding more layers and altering the network layers.

#### Defined Network
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    drop = 0.025
    self.convblock1 = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=0, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Dropout(drop),
    )

    self.convblock2 = nn.Sequential(
        nn.Conv2d(8, 16, kernel_size=3, padding=0, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Dropout(drop),
    )
    
    self.pool1 = nn.MaxPool2d(2,2)
    self.transition = nn.Sequential(
        nn.Conv2d(16, 8, kernel_size=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Dropout(drop),
    )

    self.convblock3 = nn.Sequential(
        nn.Conv2d(8, 12, kernel_size=3, padding=0, bias=False),
        nn.BatchNorm2d(12),
        nn.ReLU(),
        nn.Dropout(drop),
        nn.Conv2d(12, 16, kernel_size=3, padding=0, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Dropout(drop),
        nn.Conv2d(16, 20, kernel_size=3, padding=0, bias=False),
        nn.BatchNorm2d(20),
        nn.ReLU(),
        nn.Dropout(drop),
    )

    self.gap = nn.AvgPool2d(6)

    self.convblock4 = nn.Sequential(
        nn.Conv2d(20, 24, kernel_size=1, bias=False),
        nn.BatchNorm2d(24),
        nn.ReLU(),
        nn.Dropout(drop),
        nn.Conv2d(24, 10, kernel_size=1, bias=False),
        nn.ReLU(),
    )
  
  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.pool1(x)
    x = self.transition(x)

    x = self.convblock3(x)
    x = self.gap(x)
    x = self.convblock4(x)

    x = x.view(-1,10)
    return F.log_softmax(x, dim=-1)
```

#### Model Summary

Total params: 7,752
Trainable params: 7,752
Non-trainable params: 0

Input size (MB): 0.00
Forward/backward pass size (MB): 0.59
Params size (MB): 0.03
Estimated Total Size (MB): 0.62

#### Train and Test Accuracy obtained for diffetent epochs

EPOCH: 0
  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
Loss=0.11476197838783264 Batch_id=468 Accuracy=88.54: 100%|██████████| 469/469 [00:20<00:00, 22.51it/s]

Test set: Average loss: 0.0614, Accuracy: 9835/10000 (98.35%)

EPOCH: 1
Loss=0.0557774193584919 Batch_id=468 Accuracy=97.87: 100%|██████████| 469/469 [00:20<00:00, 22.54it/s]

Test set: Average loss: 0.0619, Accuracy: 9833/10000 (98.33%)

EPOCH: 2
Loss=0.05002601817250252 Batch_id=468 Accuracy=98.31: 100%|██████████| 469/469 [00:20<00:00, 22.41it/s]

Test set: Average loss: 0.0362, Accuracy: 9895/10000 (98.95%)

EPOCH: 3
Loss=0.12764817476272583 Batch_id=468 Accuracy=98.50: 100%|██████████| 469/469 [00:21<00:00, 22.10it/s]

Test set: Average loss: 0.0326, Accuracy: 9899/10000 (98.99%)

EPOCH: 4
Loss=0.016000008210539818 Batch_id=468 Accuracy=98.65: 100%|██████████| 469/469 [00:20<00:00, 22.38it/s]

Test set: Average loss: 0.0293, Accuracy: 9913/10000 (99.13%)

EPOCH: 5
Loss=0.04529224708676338 Batch_id=468 Accuracy=98.80: 100%|██████████| 469/469 [00:21<00:00, 22.13it/s]

Test set: Average loss: 0.0299, Accuracy: 9905/10000 (99.05%)

EPOCH: 6
Loss=0.09245741367340088 Batch_id=468 Accuracy=98.92: 100%|██████████| 469/469 [00:21<00:00, 21.90it/s]

Test set: Average loss: 0.0262, Accuracy: 9913/10000 (99.13%)

EPOCH: 7
Loss=0.10186389088630676 Batch_id=468 Accuracy=98.98: 100%|██████████| 469/469 [00:21<00:00, 21.99it/s]

Test set: Average loss: 0.0314, Accuracy: 9901/10000 (99.01%)

EPOCH: 8
Loss=0.015004717744886875 Batch_id=468 Accuracy=99.03: 100%|██████████| 469/469 [00:21<00:00, 22.18it/s]

Test set: Average loss: 0.0238, Accuracy: 9921/10000 (99.21%)

EPOCH: 9
Loss=0.04830000922083855 Batch_id=468 Accuracy=99.03: 100%|██████████| 469/469 [00:20<00:00, 22.41it/s]

Test set: Average loss: 0.0263, Accuracy: 9919/10000 (99.19%)

EPOCH: 10
Loss=0.01352918054908514 Batch_id=468 Accuracy=99.09: 100%|██████████| 469/469 [00:21<00:00, 22.04it/s]

Test set: Average loss: 0.0282, Accuracy: 9910/10000 (99.10%)

EPOCH: 11
Loss=0.005019613076001406 Batch_id=468 Accuracy=99.08: 100%|██████████| 469/469 [00:21<00:00, 22.12it/s]

Test set: Average loss: 0.0234, Accuracy: 9917/10000 (99.17%)

EPOCH: 12
Loss=0.043517131358385086 Batch_id=468 Accuracy=99.16: 100%|██████████| 469/469 [00:21<00:00, 22.30it/s]

Test set: Average loss: 0.0256, Accuracy: 9920/10000 (99.20%)

EPOCH: 13
Loss=0.007107473444193602 Batch_id=468 Accuracy=99.13: 100%|██████████| 469/469 [00:21<00:00, 22.13it/s]

Test set: Average loss: 0.0257, Accuracy: 9918/10000 (99.18%)

EPOCH: 14
Loss=0.011836051940917969 Batch_id=468 Accuracy=99.22: 100%|██████████| 469/469 [00:21<00:00, 22.05it/s]

Test set: Average loss: 0.0225, Accuracy: 9922/10000 (99.22%)


#### Resuts:
- Parameters : 7752
- Best Training Accuracy: 99.22%
- Best Test Accuracy: 99.22%

#### Analysis:
- we observe that our model is good but there is a scope of improvement as it is under fitting and hence we can increase the capacity a little more and add rotation transformation and LR Scheduler

## Step 5.
#### Target:
To increase the efficiency of the model by adding  rotation transformation and LR Scheduler. 

#### Adding Data Arguementations
``` python 
# Train Phase transformations
train_transforms = transforms.Compose([
                                     
                                       transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])

# Test Phase transformations
test_transforms = transforms.Compose([
                                     
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])
```
### Defined Network 
The network is same as the previous network 


#### Train and Test Accuracy obtained for diffetent epochs
EPOCH: 1
  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
Loss=0.38077983260154724 Batch_id=468 Accuracy=62.51: 100%|██████████| 469/469 [00:22<00:00, 21.29it/s]

Test set: Average loss: 0.4338, Accuracy: 8696/10000 (86.96%)

EPOCH: 2
Loss=0.18096864223480225 Batch_id=468 Accuracy=90.63: 100%|██████████| 469/469 [00:22<00:00, 21.08it/s]

Test set: Average loss: 0.0811, Accuracy: 9795/10000 (97.95%)

EPOCH: 3
Loss=0.09088011831045151 Batch_id=468 Accuracy=97.09: 100%|██████████| 469/469 [00:22<00:00, 21.24it/s]

Test set: Average loss: 0.0694, Accuracy: 9802/10000 (98.02%)

EPOCH: 4
Loss=0.0809013620018959 Batch_id=468 Accuracy=97.68: 100%|██████████| 469/469 [00:21<00:00, 21.98it/s]

Test set: Average loss: 0.0452, Accuracy: 9860/10000 (98.60%)

EPOCH: 5
Loss=0.03515147417783737 Batch_id=468 Accuracy=98.04: 100%|██████████| 469/469 [00:21<00:00, 21.93it/s]

Test set: Average loss: 0.0320, Accuracy: 9890/10000 (98.90%)

EPOCH: 6
Loss=0.01072074007242918 Batch_id=468 Accuracy=98.24: 100%|██████████| 469/469 [00:21<00:00, 21.71it/s]

Test set: Average loss: 0.0261, Accuracy: 9927/10000 (99.27%)

EPOCH: 7
Loss=0.01881556585431099 Batch_id=468 Accuracy=98.51: 100%|██████████| 469/469 [00:21<00:00, 21.70it/s]

Test set: Average loss: 0.0318, Accuracy: 9904/10000 (99.04%)

EPOCH: 8
Loss=0.017045551910996437 Batch_id=468 Accuracy=98.65: 100%|██████████| 469/469 [00:21<00:00, 21.81it/s]

Test set: Average loss: 0.0291, Accuracy: 9906/10000 (99.06%)

EPOCH: 9
Loss=0.17183703184127808 Batch_id=468 Accuracy=98.75: 100%|██████████| 469/469 [00:21<00:00, 21.88it/s]

Test set: Average loss: 0.0242, Accuracy: 9931/10000 (99.31%)

EPOCH: 10
Loss=0.014050036668777466 Batch_id=468 Accuracy=98.82: 100%|██████████| 469/469 [00:21<00:00, 22.01it/s]

Test set: Average loss: 0.0204, Accuracy: 9936/10000 (99.36%)

EPOCH: 11
Loss=0.007068320643156767 Batch_id=468 Accuracy=98.95: 100%|██████████| 469/469 [00:21<00:00, 21.75it/s]

Test set: Average loss: 0.0193, Accuracy: 9946/10000 (99.46%)

EPOCH: 12
Loss=0.009403491392731667 Batch_id=468 Accuracy=98.98: 100%|██████████| 469/469 [00:22<00:00, 21.31it/s]

Test set: Average loss: 0.0204, Accuracy: 9934/10000 (99.34%)

EPOCH: 13
Loss=0.04482561722397804 Batch_id=468 Accuracy=99.02: 100%|██████████| 469/469 [00:21<00:00, 21.55it/s]

Test set: Average loss: 0.0177, Accuracy: 9945/10000 (99.45%)

EPOCH: 14
Loss=0.019093163311481476 Batch_id=468 Accuracy=99.12: 100%|██████████| 469/469 [00:22<00:00, 21.26it/s]

Test set: Average loss: 0.0169, Accuracy: 9951/10000 (99.51%)

EPOCH: 15
Loss=0.043417736887931824 Batch_id=468 Accuracy=99.18: 100%|██████████| 469/469 [00:21<00:00, 21.54it/s]

Test set: Average loss: 0.0165, Accuracy: 9955/10000 (99.55%)

#### Resuts:
- Parameters : 7752
- Best Training Accuracy: 99.18%
- Best Test Accuracy: 99.55%

This model is a good model as we have reached an accuracy of 99.55% with less than 10K parameters. 

## Model Comparision Table
---------------------
| Model | Statistics | Analysis|
|---| ---|---|
|Model 1: The Basic Skeleton | <Li> Total Parameters : 7.3k|
                               <Li> Training Acc : 98.58%
                               <Li> Training Loss : 0.01337
                               <Li> Test Acc : 98.66%
                               <Li> Test Loss : 0.0464 | have
|---|---|---|
