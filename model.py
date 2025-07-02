import torch.nn as nn
import math
import torch

'''
Wrappers for convolution layers with different filter lengths

Parameters
------------
in_planes: int, Dimensionality of inputs to the convolution layer, default None
out_planes: int, Dimensionality of outputs of the convolution layer , default None
'''
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)



class BasicBlock3x3(nn.Module):
    '''
    Convolutional layer when using a 3x3 filter.  Uses 3x3 1d convolutions follwed by ReLU and batch normalization
    '''
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #optional residual connection, not used in paper results
        #out += residual
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    '''
    Convolutional layer when using a 5x5 filter.  Uses 5x5 1d convolutions follwed by ReLU and batch normalization
    '''
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:,:,0:-d] + out
        out1 = self.relu(out1)

        #optional residual connection, not used in paper results
        # out += residual

        return out1



class BasicBlock7x7(nn.Module):
    '''
    Convolutional layer when using a 7x7 filter.  Uses 7x7 1d convolutions follwed by ReLU and batch normalization
    '''
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)

        #optional residual connection, not used in paper results
        # out += residual

        return out1




class tcn(nn.Module):
    '''
    Temporal convolution network that uses Class Activation Maps(CAMs) for interpretability.  The model consists of repeated
    1d convolutions followed by a global average pooling layer and a fully connected classification layer. The outputs of the
    convolutional layers and weights of the fully connected layer are used to calculate the CAMs.
    '''
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=2):
        '''
        Initializer for a TCN object. Note that TCN is a subclass of torch.nn.Module.

        Parameters
        ------------
        input_channel: int, Dimensionality of inputs to the MLP, default None
        layers: int, Number of convolutions per convolutional layer, default [1,1,1,1]
        num_classes: int, Number of output classes, default 2
        '''
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(tcn, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=3, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=1)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 32, layers[1], stride=1)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 16, layers[2], stride=1)

        # maxpooling kernel size: 16, 11, 6
        self.deconvz = nn.ConvTranspose1d(16, 16, 10, stride=3)
        self.num_classes = num_classes
        self.fc = nn.Linear(16, num_classes)

        # optional, not used in paper results: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))
        layers.append(nn.Dropout(.2))
        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        layers.append(nn.Dropout(.2))
        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        layers.append(nn.Dropout(.2))
        return nn.Sequential(*layers)

    def forward(self, x0, generateCAMs = False):   
        '''
        Forward pass function for TCN 

        Parameters: 
        ------------
        x0: torch.Tensor, shape: (num_seq, sequence_length, input_channel), Flattened batch of inputs

        Returns: 
        ------------
        out: torch.Tensor, shape: (num_seq, num_classes),Flattened batch of outputs
        '''
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        y = self.deconvz(x)

        out = y
        out = out.mean(2)
       
        out = self.fc(out)
        y = torch.permute(y, [0,2,1])
        cams = torch.zeros([len(y),self.num_classes,len(y[0]),len(self.fc.weight[0])])

        if generateCAMs:
            # for i in range(len(y)):
            #     for j in range(len(y[0])):
            #         for k in range(self.num_classes):
            #             cams[i,k,j,:] = self.fc.weight[k,:] * y[i,j,:]
            cams = self.fc.weight[None, :, None, :] * y[:, None, :, :]
            cams = cams.mean(3)
            cams.unsqueeze(3)
        return out, cams