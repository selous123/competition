import os
import csv
import torch
from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
class CompetetionDataset(data.Dataset):
    def __init__(self, root, transform = None):
        self.imageNames = os.listdir(root)
        self.imgDir = root
        self.transform = transform

    def __getitem__(self, index):
        imageName = self.imageNames[index]
        imagePath = os.path.join(self.imgDir, imageName)
        img = Image.open(imagePath)
        if self.transform is not None:
            img = self.transform(img)
        return img, imageName

    def __len__(self):
        return len(self.imageNames)

modelStruc = "resnetAttn"
testBatchsize = 1
modelPath = "/home/lrh/program/git/pytorch-example/competition_nh/models/resnetAttn/resnetAttn_200_compe_99.97.pth"
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
])
testRoot = "/home/lrh/dataset/competition_nh/norm_test"
output_csv_path = "/home/lrh/program/git/pytorch-example/competition_nh/test.csv"
testDataset = CompetetionDataset(testRoot, transforms_test)
testloader = torch.utils.data.DataLoader(testDataset, batch_size=testBatchsize, shuffle=False, num_workers=8)

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
    optimizer = optim.Adam(params_to_update, lr=learningRate, betas=(0.5,0.999))
elif modelStruc == "vgg19Attn":
    clf = AttnVGG(num_classes=200, attention=True, normalize_attn=True)

elif modelStruc == "resnetAttn":
    from resnetAttn.residual_attention_network import ResidualAttentionModel_56 as ResidualAttentionModel
    clf = ResidualAttentionModel()
    #lr = 0.1
    #optimizer = optim.Adam(clf.parameters(), lr=learningRate, betas=(0.5,0.999), weight_decay=0.0001)
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
clf.load_state_dict(torch.load(modelPath))
clf.eval()

def test():

    file_test = open(output_csv_path, 'w', newline='')
    csv_writer = csv.writer(file_test)
    csv_writer.writerow(["ImageName", "CategoryId"])

    for batchItr, (images, imageNames) in enumerate(testloader):
        images = images.cuda()
        outputs = clf(images)
        predict = outputs.argmax(dim = 1)
        csv_writer.writerow([imageNames[0], predict[0].data.cpu().numpy() + 1])
        print(batchItr)


if __name__=="__main__":
    test()
