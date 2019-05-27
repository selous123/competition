import torch.utils.data as data
import pandas
import os
from PIL import Image
import torch
from torchvision.transforms import transforms
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
from libadver.utils import *
import torch.backends.cudnn as cudnn
from networks import AttnVGG
import torch.nn.init as init

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)



class CompetetionDataset(data.Dataset):
    def __init__(self, root, imgDir, transform = None):
        data_pd = pandas.read_csv(root)
        self.imageNames = data_pd["ImageName"]
        self.labels = data_pd["CategoryId"]
        self.imgDir = imgDir
        self.transform = transform

    def __getitem__(self, index):
        label = self.labels[index]
        imageName = self.imageNames[index]
        imagePath = os.path.join(self.imgDir, imageName)
        img = Image.open(imagePath)
        if self.transform is not None:
            img = self.transform(img)
        return img, label - 1

    def __len__(self):
        return len(self.imageNames)

trainRoot = "/home/lrh/dataset/competition_nh/data.csv"
valRoot = "/home/lrh/dataset/competition_nh/val.csv"
imgDir = "/home/lrh/dataset/competition_nh/norm_data"
trainBatchsize = 30
valBatchsize = 4
modelStruc = "resnetAttn"
modelPath = "/home/lrh/program/git/pytorch-example/competition_nh/models/resnetAttn/resnetAttn_200_compe_99.81.pth"

epochNum = 100
learningRate = 2e-7
feature_extracted = False
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
transforms_train = transforms.Compose([
        #transforms.Resize([256,256]),
        #transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((224, 224), padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
])

transforms_val = transforms.Compose([
        #transforms.Resize([256,256]),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
])

trainDataset = CompetetionDataset(trainRoot,imgDir,transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=trainBatchsize, shuffle=True, num_workers=4)

valDataset = CompetetionDataset(valRoot,imgDir,transform=transforms_val)
valoader = torch.utils.data.DataLoader(valDataset, batch_size=valBatchsize, shuffle=True, num_workers=4)


if modelStruc == "densenet161":
    clf = models.densenet161(pretrained = True)
    if feature_extracted:
        for param in clf.parameters():
            param.requires_grad = False
    fc = nn.Sequential(
        nn.Linear(2208,1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024,200),
    )
    clf.classifier = fc
    optimizer = optim.Adam(clf.parameters(), lr=learningRate, betas=(0.5,0.999))
elif modelStruc == "vgg19Attn":
    clf = AttnVGG(num_classes=200, attention=True, normalize_attn=True)

elif modelStruc == "resnetAttn":
    from resnetAttn.residual_attention_network import ResidualAttentionModel_56 as ResidualAttentionModel
    clf = ResidualAttentionModel()
    #lr = 0.1
    #clf.apply(init_params)
    optimizer = optim.Adam(clf.parameters(), lr=learningRate, betas=(0.5,0.999), weight_decay=0.0001)
    #optimizer = optim.SGD(clf.parameters(), lr=learningRate, momentum=0.9, nesterov=True, weight_decay=0.0001)
elif modelStruc == "resnext101":
    clf = models.resnext101_32x8d(pretrained=True)
    fc = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,200)
        )
    clf.fc = fc

clf = clf.cuda()
clf = torch.nn.DataParallel(clf)
cudnn.benchmark = True

if modelStruc == "resnetAttn":
    clf.load_state_dict(torch.load(modelPath))

# params_to_update = []
# for name,param in clf.named_parameters():
#     if param.requires_grad == True:
#         params_to_update.append(param)
#         print("\t",name)


criterion = nn.CrossEntropyLoss()


def train(epoch):
    clf.train()
    print("train Epoch : %d\n" %epoch)
    total = 0
    correct = 0
    for batchItr, (images, labels) in enumerate(trainloader):
        images, labels = images.cuda(), labels.cuda()
        outputs = clf(images)
        if modelStruc == "vgg19Attn":
            outputs, _, _ = outputs
        #torch.softmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predict = outputs.argmax(dim = 1)
        total = total + images.size(0)
        correct = correct + labels.eq(predict).sum()
        #print(correct)

        progress_bar(batchItr, len(trainloader), "loss:%.3f, ACC:%.2f%%" %(loss, 100.0 * float(correct) / total))


def val(epoch):
    clf.eval()
    print("validation Epoch : %d\n" %epoch)
    total = 0
    correct = 0
    for batchItr, (images, labels) in enumerate(valoader):
        images, labels = images.cuda(), labels.cuda()
        outputs = clf(images)
        if modelStruc == "vgg19Attn":
            outputs, _, _ = outputs
        predict = outputs.argmax(dim = 1)
        total = total + images.size(0)
        correct = correct + labels.eq(predict).sum()
        #print(correct)
        progress_bar(batchItr, len(valoader), "ACC:%.2f%%" %(100.0 * float(correct) / total))

    acc = 100.0 * float(correct) / total
    return acc


if __name__=="__main__":
    bestAcc = 0
    for epoch in range(epochNum):

        if epoch == 5:
            optimizer = optim.Adam(clf.parameters(), lr=1e-7, betas=(0.5,0.999), weight_decay=0.0001)
        elif epoch == 10:
            optimizer = optim.Adam(clf.parameters(), lr=2e-8, betas=(0.5,0.999), weight_decay=0.0001)
        elif epoch == 15:
            optimizer = optim.Adam(clf.parameters(), lr=1e-8, betas=(0.5,0.999), weight_decay=0.0001)

        train(epoch)
        acc = val(epoch)
        if bestAcc < acc:
            bestAcc = acc
            torch.save(clf.state_dict(), "models/resnetAttn/resnetAttn_200_compe_%.2f.pth" %bestAcc)
            print("bestAcc: %.2f", bestAcc)
