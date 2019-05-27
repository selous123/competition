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
from senet.se_resnet import se_resnet50

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
trainBatchsize = 16
valBatchsize = 4

epochNum = 20
learningRate = 2e-5
feature_extracted = False
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
transforms_train = transforms.Compose([
        #transforms.Resize([256,256]),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
])

trainDataset = CompetetionDataset(trainRoot,imgDir,transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=trainBatchsize, shuffle=True, num_workers=8)

valDataset = CompetetionDataset(valRoot,imgDir,transform=transforms_train)
valoader = torch.utils.data.DataLoader(valDataset, batch_size=valBatchsize, shuffle=True, num_workers=8)

#clf = models.densenet161(pretrained = True)
#clf = models.resnet50(pretrained = True)
clf = se_resnet50(num_classes = 1000, pretrained = True)
if feature_extracted:
    for param in clf.parameters():
        param.requires_grad = False

fc = nn.Sequential(
        nn.Linear(2048,1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024,1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024,200)
    )
clf.fc = fc
# fc = nn.Sequential(
#     nn.Linear(2208,1024),
#     nn.ReLU(inplace=True),
#     nn.Linear(1024,200),
# )
#clf.classifier = fc
clf = clf.cuda()
clf = torch.nn.DataParallel(clf)
cudnn.benchmark = True


params_to_update = []
for name,param in clf.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params_to_update, lr=learningRate, betas=(0.5,0.999))


def train(epoch):
    clf.train()
    print("train Epoch : %d\n" %epoch)
    total = 0
    correct = 0
    for batchItr, (images, labels) in enumerate(trainloader):
        images, labels = images.cuda(), labels.cuda()
        outputs = clf(images)
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
        train(epoch)
        acc = val(epoch)
        if bestAcc < acc:
            bestAcc = acc
            torch.save(clf.state_dict(), "models/seresnet50_200_compe.pth")
            print("bestAcc: %.2f", bestAcc)
