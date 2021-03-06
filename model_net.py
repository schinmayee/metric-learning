import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import init

class SimpleNet(nn.Module):
    def __init__(self, feature_size=64, im_size=128, normalize=False):
        super(SimpleNet, self).__init__()
        self.normalize = normalize
        self.feature_size=feature_size
        self.im_size = im_size
        self.h1_len = (self.im_size-4)/2
        self.h2_len = (self.h1_len-4)/2
        self.fc1_len = self.h2_len*self.h2_len*20
        # all the layers
        self.bn0   = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.bn1   = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2   = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d(p=0.3)
        self.fc1 = nn.Linear(self.fc1_len, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, feature_size)
	init.xavier_normal(self.conv1.weight)
	init.xavier_normal(self.conv2.weight)
	init.xavier_normal(self.fc1.weight)
	init.xavier_normal(self.fc2.weight)

    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), 2))
        x = x.view(-1, self.fc1_len)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if self.normalize:
            return x/torch.norm(x,2,1).repeat(1, self.feature_size)
        else:
            return x

    def SetLearningRate(self, lr1, lr2):
	print('Setting learning rate for simple net')
        d = [{ 'params' : self.parameters(), 'lr': lr2 }]
        return d

class ShallowNet(nn.Module):
    def __init__(self, feature_size=64, im_size=96, normalize=False):
        super(ShallowNet, self).__init__()
        self.normalize = normalize
        self.feature_size=feature_size
        self.im_size = im_size
        self.h1_len = (self.im_size-6)/2
        self.h2_len = (self.h1_len-5)/2
        self.fc1_len = self.h2_len*self.h2_len*32
        # all the layers
        self.bn0   = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7)
        self.conv1_drop = nn.Dropout2d(p=0.3)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv2_drop = nn.Dropout2d(p=0.3)
        self.fc1 = nn.Linear(self.fc1_len, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, feature_size)
	init.xavier_normal(self.conv1.weight)
	init.xavier_normal(self.conv2.weight)
	init.xavier_normal(self.fc1.weight)
	init.xavier_normal(self.fc2.weight)

    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(F.max_pool2d(self.conv1_drop(self.bn1(self.conv1(x))), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), 2))
        x = x.view(-1, self.fc1_len)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if self.normalize:
            return x/torch.norm(x,2,1).repeat(1, self.feature_size)
        else:
            return x

    def SetLearningRate(self, lr1, lr2):
	print('Setting learning rate for shallow net')
        d = [{ 'params' : self.parameters(), 'lr': lr2 }]
        return d

class InceptionBased(nn.Module):
    def __init__(self, feature_size=2048, im_size=299, normalize=False):
        super(InceptionBased, self).__init__()
        self.normalize = normalize
        self.im_size = 299
        self.feature_size=feature_size
        self.inception = torchvision.models.inception_v3(pretrained=True)
        self.inception.fc = nn.Linear(2048, feature_size)
	init.xavier_normal(self.inception.fc.weight)

    def forward(self, x):
        #y = self.inception(x)
        ## weird result in training mode, probably a bug in inception module?
        #if self.training:
        #    if self.normalize:
        #        return y[0]/torch.norm(y[0],2,1).repeat(1, self.feature_size)
        #    else:
        #        return y[0]
        #else:
        #    if self.normalize:
        #        return y/torch.norm(y,2,1).repeat(1, self.feature_size)
        #    else:
        #        return y
        if self.inception.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.inception.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.inception.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.inception.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.inception.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.inception.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.inception.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.inception.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.inception.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.inception.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6e(x)
        # 17 x 17 x 768
        if self.inception.training and self.inception.aux_logits:
            aux = self.inception.AuxLogits(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.inception.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.inception.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
	# x = x.view(-1, self.feature_size)

	# 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.inception.fc(x)
	if self.normalize:
        	return x/torch.norm(x,2,1).repeat(1, self.feature_size)
	else:
		return x

    def SetLearningRate(self, lr1, lr2):
	print('Setting learning rate for inception net')
        d = [
                { 'params' : self.inception.Conv2d_1a_3x3.parameters(), 'lr': lr1 },
                { 'params' : self.inception.Conv2d_2a_3x3.parameters(), 'lr': lr1 },
                { 'params' : self.inception.Conv2d_2b_3x3.parameters(), 'lr': lr1 },
                { 'params' : self.inception.Conv2d_3b_1x1.parameters(), 'lr': lr1 },
                { 'params' : self.inception.Conv2d_4a_3x3.parameters(), 'lr': lr1 },
                { 'params' : self.inception.Mixed_5b.parameters(), 'lr': 2*lr1 },
                { 'params' : self.inception.Mixed_5c.parameters(), 'lr': 2*lr1 },
                { 'params' : self.inception.Mixed_5d.parameters(), 'lr': 2*lr1 },
                { 'params' : self.inception.Mixed_6a.parameters(), 'lr': 2*lr1 },
                { 'params' : self.inception.Mixed_6b.parameters(), 'lr': 2*lr1 },
                { 'params' : self.inception.Mixed_6c.parameters(), 'lr': 2*lr1 },
                { 'params' : self.inception.Mixed_6d.parameters(), 'lr': 2*lr1 },
                { 'params' : self.inception.Mixed_6e.parameters(), 'lr': 2*lr1 },
                { 'params' : self.inception.AuxLogits.parameters(), 'lr': 2*lr1 },
                { 'params' : self.inception.Mixed_7a.parameters(), 'lr': 2*lr1 },
                { 'params' : self.inception.Mixed_7b.parameters(), 'lr': 2*lr1 },
                { 'params' : self.inception.Mixed_7c.parameters(), 'lr': 2*lr1 },
                { 'params' : self.inception.fc.parameters(), 'lr': lr2 },
            ]
        return d

class SqueezeNetBased(nn.Module):
    def __init__(self, feature_size=64, im_size=224, normalize=False):
        super(SqueezeNetBased, self).__init__()
        self.normalize = normalize
        self.im_size = 224
        self.feature_size = feature_size
        self.features = torchvision.models.squeezenet1_1(pretrained=True).features
        final_conv = nn.Conv2d(512, feature_size, kernel_size=1)
        classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13)
        )
        self.classifier = classifier
	init.xavier_normal(final_conv.weight)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(-1, self.feature_size)
        if self.normalize:
            return x/torch.norm(x,2,1).repeat(1, self.feature_size)
        else:
            return x

    def SetLearningRate(self, lr1, lr2):
	print('Setting learning rate for squeeze net')
        d = [
                { 'params' : self.features.parameters(), 'lr': lr1 },
                { 'params' : self.classifier.parameters(), 'lr': lr2 },
            ]
        return d

class ResNetBased(nn.Module):
    def __init__(self, feature_size=64, im_size=224, normalize=False):
        super(ResNetBased, self).__init__()
        self.normalize = normalize
        self.im_size = 224
        self.feature_size = feature_size
        self.resnet = torchvision.models.resnet50(pretrained=True)
        fc = nn.Linear(2048, feature_size)
        self.resnet.fc = fc
	init.xavier_normal(self.resnet.fc.weight)

    def forward(self, x):
        x = self.resnet(x)
        if self.normalize:
            return x/torch.norm(x,2,1).repeat(1, self.feature_size)
        else:
            return x

    def SetLearningRate(self, lr1, lr2):
	print('Setting learning rate for resnet')
        d = [
                { 'params' : self.resnet.conv1.parameters(), 'lr': lr1 },
                { 'params' : self.resnet.bn1.parameters(), 'lr': lr1 },
                { 'params' : self.resnet.layer1.parameters(), 'lr': lr1 },
                { 'params' : self.resnet.layer2.parameters(), 'lr': lr1 },
                { 'params' : self.resnet.layer3.parameters(), 'lr': lr1 },
                { 'params' : self.resnet.layer4.parameters(), 'lr': lr1 },
                { 'params' : self.resnet.fc.parameters(), 'lr': lr2 },
            ]
        return d
