import torch.nn as nn

class UnidadConvolucional(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UnidadConvolucional,self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class ConvolucionalBebeAudioLite(nn.Module):
    def __init__(self):
        super(ConvolucionalBebeAudioLite,self).__init__()
        
        self.unit1 = UnidadConvolucional(in_channels=3,out_channels=128)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.unit2 = UnidadConvolucional(in_channels=128, out_channels=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.unit3 = UnidadConvolucional(in_channels=64, out_channels=32)
        self.pool3 = nn.AvgPool2d(kernel_size=4)
        
        self.net = nn.Sequential(self.unit1, self.pool1, self.unit2, self.pool2, self.unit3, self.pool3)

        self.fc = nn.Linear(in_features=20000,out_features=2)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,20000)
        output = self.fc(output)
        return output

class ConvolucionalBebeAudioLiteV2(nn.Module):
    def __init__(self):
        super(ConvolucionalBebeAudioLiteV2,self).__init__()
        
        self.unit1 = UnidadConvolucional(in_channels=3,out_channels=128)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.unit2 = UnidadConvolucional(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.unit3 = UnidadConvolucional(in_channels=128, out_channels=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.unit4 = UnidadConvolucional(in_channels=64, out_channels=32)
        self.pool4 = nn.AvgPool2d(kernel_size=4)
        
        self.net = nn.Sequential(self.unit1, self.pool1, self.unit2, self.pool2, self.unit3, self.pool3, self.unit4, self.pool4)

        self.fc = nn.Linear(in_features=4608,out_features=2)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,4608)
        output = self.fc(output)
        return output


class ConvolucionalBebeAudioFull(nn.Module):
    def __init__(self):
        super(ConvolucionalBebeAudioFull,self).__init__()
        
        self.unit1 = UnidadConvolucional(in_channels=3,out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit2 = UnidadConvolucional(in_channels=64, out_channels=64)
        self.unit3 = UnidadConvolucional(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = UnidadConvolucional(in_channels=64, out_channels=64)
        self.unit5 = UnidadConvolucional(in_channels=64, out_channels=64)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit6 = UnidadConvolucional(in_channels=64, out_channels=64)
        self.unit7 = UnidadConvolucional(in_channels=64, out_channels=64)

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        
        self.net = nn.Sequential(self.unit1, self.pool1, self.unit2, self.unit3, self.pool2, self.unit4, self.unit5, self.pool3, self.unit6
                                 ,self.unit7, self.avgpool)

        self.fc = nn.Linear(in_features=9216,out_features=2)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,9216)
        output = self.fc(output)
        return output