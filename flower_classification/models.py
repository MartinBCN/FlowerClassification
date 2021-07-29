from typing import List
import torch.nn as nn
import torch.nn.functional as F


class Solution(nn.Module):

    def __init__(self):
        super(Solution, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # obejctive is to bring down the image size to single unit-->
        # here given image size is 224x224px
        self.conv1 = nn.Conv2d(1, 32, 5)
        # 224--> 224-5+1=220
        self.pool1 = nn.MaxPool2d(2, 2)
        # 220/2=110 ...(32,110,110)

        self.conv2 = nn.Conv2d(32, 64, 3)
        # 110--> 110-3+1=108
        self.pool2 = nn.MaxPool2d(2, 2)
        # 108/2=54
        self.batch_norm_2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        # 54-->54-3+1=52
        self.pool3 = nn.MaxPool2d(2, 2)
        # 52/2=26

        self.conv4 = nn.Conv2d(128, 256, 3)
        # 26-->26-3+1=24
        self.pool4 = nn.MaxPool2d(2, 2)
        # 24/2=12
        self.batch_norm_4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, 1)
        # 12-->12-1+1=12
        self.pool5 = nn.MaxPool2d(2, 2)
        # 12/2=6

        # 6x6x512
        self.fc1 = nn.Linear(6 * 6 * 512, 1024)
        #         self.fc2 = nn.Linear(1024,1024)
        self.fc2 = nn.Linear(1024, 136)

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)
        # self.fc2_drop = nn.Dropout(p=.5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.batch_norm_2(self.pool2(F.relu(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = self.drop4(self.batch_norm_4(self.pool4(F.relu(self.conv4(x)))))
        x = self.drop5(self.pool5(F.relu(self.conv5(x))))
        x = x.view(batch_size, -1)
        x = self.drop6(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = x.view(batch_size, -1, 2)

        # a modified x, having gone through all the layers of your model, should be returned
        return x


class Final(nn.Module):

    def __init__(self):

        super(Final, self).__init__()

        self.convolution_layers = nn.Sequential(
            # First Layer: 32 5x5 Filter, Max-Pooling 2x2
            # Outputsize: (224 - 5 + 1)/2 = 220/2 = 110
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.1),

            # Second Layer: 64 3x3 Filter, Max-Pooling 2x2
            # Outputsize: (110 - 3 + 1)/2 = 108/2 = 54
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.2),

            # Third Layer: 128 3x3 Filter, Max-Pooling 2x2
            # Outputsize: (54 - 3 + 1)/2 = 52/2 = 26
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.3),

            # Fourth Layer: 256 3x3 Filter, Max-Pooling 2x2
            # Outputsize: (26 - 3 + 1)/2 = 24/2 = 12
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.4),

            # Fourth Layer: 512 1x1 Filter, Max-Pooling 2x2
            # Outputsize: (12 - 1 + 1)/2 = 6
            nn.Conv2d(256, 512, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.5),
        )

        self.linear_layers = nn.Sequential(

            nn.Linear(6 * 6 * 512, 1024),
            nn.ReLU(),
            # nn.Dropout(p=0.6),
            # nn.Linear(2056, 1024),
            # nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(1024, 136)

        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.convolution_layers(x)
        x = x.view(batch_size, -1)
        x = self.linear_layers(x)
        x = x.view(batch_size, -1, 2)

        return x


class Net(nn.Module):

    def __init__(self, image_size: int, filter_size: List[int], feature_maps: List[int],
                 second_linear_layer: bool = True):

        super(Net, self).__init__()
        self.image_size = image_size
        self.filter_size = filter_size
        self.features = feature_maps
        self.input_size = [1] + feature_maps[:-1]

        output_size = 224

        conv_layers = []
        dropout_prob = 0.1
        for input, features, filter in zip(self.input_size, self.features, self.filter_size):

            output_size = int((output_size - filter + 1) / 2)
            conv_layers.append(nn.Conv2d(input, features, filter))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm2d(features))
            conv_layers.append(nn.MaxPool2d(2, 2),)
            conv_layers.append(nn.Dropout(p=dropout_prob))
            dropout_prob += 0.1

        self.convolution_layers = nn.Sequential(*conv_layers)

        if second_linear_layer:

            self.linear_layers = nn.Sequential(

                nn.Linear(output_size ** 2 * self.features[-1], 1024),
                # nn.Linear(output_size**2 * self.features[-1], 2056),
                # nn.ReLU(),
                # nn.BatchNorm1d(2056),
                # nn.Dropout(p=0.6),
                #
                # nn.Linear(2056, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.6),
                nn.Linear(1024, 136)

            )

        else:
            self.linear_layers = nn.Linear(output_size**2 * self.features[-1], 136)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.convolution_layers(x)
        x = x.view(batch_size, -1)
        x = self.linear_layers(x)
        x = x.view(batch_size, -1, 2)

        return x


if __name__ == '__main__':

    import torch
    import pandas as pd
    from torch.optim import Adam

    parameter = []
    forward_time = []
    backward_time = []
    feature_maps = []
    filter_sizes = []
    second_linear = []

    models = []
    models.append({'Filter Sizes': [3], 'Feature Maps': [128], 'Second Linear Layer': True})
    models.append({'Filter Sizes': [3], 'Feature Maps': [128], 'Second Linear Layer': False})

    models.append({'Filter Sizes': [5, 3], 'Feature Maps': [32, 64], 'Second Linear Layer': True})
    models.append({'Filter Sizes': [5, 3], 'Feature Maps': [32, 64], 'Second Linear Layer': False})

    models.append({'Filter Sizes': [3, 1], 'Feature Maps': [32, 64], 'Second Linear Layer': True})
    models.append({'Filter Sizes': [3, 1], 'Feature Maps': [32, 64], 'Second Linear Layer': False})

    models.append({'Filter Sizes': [5, 3, 1], 'Feature Maps': [16, 32, 64], 'Second Linear Layer': True})
    models.append({'Filter Sizes': [5, 3, 1], 'Feature Maps': [16, 32, 64], 'Second Linear Layer': False})

    models.append({'Filter Sizes': [5, 3, 3, 1], 'Feature Maps': [16, 32, 64, 128], 'Second Linear Layer': True})
    models.append({'Filter Sizes': [5, 3, 3, 1], 'Feature Maps': [16, 32, 64, 128], 'Second Linear Layer': False})

    models.append({'Filter Sizes': [5, 5, 3, 3, 1], 'Feature Maps': [16, 32, 64, 128, 256],
                   'Second Linear Layer': True})
    models.append({'Filter Sizes': [5, 5, 3, 3, 1], 'Feature Maps': [16, 32, 64, 128, 256],
                   'Second Linear Layer': False})

    models.append({'Filter Sizes': [5, 5, 3, 3, 1], 'Feature Maps': [32, 64, 128, 256, 512],
                   'Second Linear Layer': True})
    models.append({'Filter Sizes': [5, 5, 3, 3, 1], 'Feature Maps': [32, 64, 128, 256, 512],
                   'Second Linear Layer': False})


    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    for model in models:

        # --- Instantiate ---
        net = Net(image_size=224, filter_size=model['Filter Sizes'], feature_maps=model['Feature Maps'],
                       second_linear_layer=model['Second Linear Layer'])
        criterion = nn.MSELoss()
        optimizer = Adam(net.parameters())

        n_param = sum(p.numel() for p in net.parameters())
        parameter.append(int(n_param / 1e6))

        # --- Forward Step ---
        t = torch.rand([20, 1, 224, 224])
        start = pd.Timestamp.now()
        output = net(t)
        time_delta = pd.Timedelta(pd.Timestamp.now() - start).microseconds / 1000
        assert output.shape == (20, 68, 2)
        forward_time.append(time_delta)

        # --- Backward Step ---
        start = pd.Timestamp.now()
        loss = criterion(output, torch.rand([20, 68, 2]))
        optimizer.step()
        time_delta = pd.Timedelta(pd.Timestamp.now() - start).microseconds / 1000
        backward_time.append(time_delta)

        feature_maps.append(model['Feature Maps'])
        filter_sizes.append(model['Filter Sizes'])
        second_linear.append(model['Second Linear Layer'])

    df = pd.DataFrame({'Feature Maps': feature_maps, 'Filter Sizes': filter_sizes, '# Parameter [Mio]': parameter,
                       'Second Linear Layer': second_linear,
                       'Forward Time [ms]': forward_time, 'Backward Time [ms]': backward_time})
    print(df)
