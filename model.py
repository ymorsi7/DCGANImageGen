import torch
import torch.nn as nn

class GeneratorGAN(nn.Module):
    '''
    Defines the generator architecture
    Inputs: kernel size, padding value
    Outputs: 3 x 64 x 64 images
    '''
    def __init__(self, kernSize, padVal):
        super(GeneratorGAN, self).__init__()
        self.conv1 = nn.ConvTranspose2d(100, 1024, kernel_size = 4, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU(inplace = True)

        self.conv2 = nn.ConvTranspose2d(1024, 512, kernel_size = kernSize, stride = 2, padding = padVal, output_padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace = True)
        self.conv3 = nn.ConvTranspose2d(512, 256, kernel_size = kernSize, stride = 2, padding = padVal, output_padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace = True)
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size = kernSize, stride = 2, padding = padVal, output_padding = 1, bias = False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace = True)
        self.conv5 = nn.ConvTranspose2d(128, 3, kernel_size = kernSize, stride = 2, padding = padVal, output_padding = 1, bias = False)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        fc1 = self.relu1(self.bn1(self.conv1(x)))
        fc2 = self.relu2(self.bn2(self.conv2(fc1)))
        fc3 = self.relu3(self.bn3(self.conv3(fc2)))
        fc4 = self.relu4(self.bn4(self.conv4(fc3)))
        return self.tanh(self.conv5(fc4))


class DiscriminatorGAN(nn.Module):
    '''
    Defines the Discriminator Architecture
    Inputs: kernel size, padding
    Outputs: Scalar of probability of realisticity of image
    '''
    def __init__(self, kernSize, padVal):
        super(DiscriminatorGAN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = kernSize, stride = 2, padding = padVal,  bias = False)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace = True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = kernSize, stride = 2, padding = padVal, bias = False)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace = True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = kernSize, stride = 2, padding = padVal, bias = False)
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size = kernSize, stride = 2, padding = padVal, bias = False)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace = True)
        self.conv5 = nn.Conv2d(512, 1, kernel_size = 4, stride = 1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fc1 = self.lrelu1(self.conv1(x))
        fc2 = self.lrelu2(self.bn2(self.conv2(fc1)))
        fc3 = self.lrelu3(self.bn3(self.conv3(fc2)))
        fc4 = self.lrelu4(self.bn4(self.conv4(fc3)))
        return self.sigmoid(self.conv5(fc4))
    