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

testBatchsize = 1
modelPath = "/home/lrh/program/git/pytorch-example/competition_nh/models/seresnet50_200_compe.pth"
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
transforms_train = transforms.Compose([
        #transforms.Resize([256,256]),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
])
testRoot = "/home/lrh/dataset/competition_nh/norm_test"
output_csv_path = "/home/lrh/program/git/pytorch-example/competition_nh/test.csv"
testDataset = CompetetionDataset(testRoot, transforms_train)
testloader = torch.utils.data.DataLoader(testDataset, batch_size=testBatchsize, shuffle=False, num_workers=8)

from senet.se_resnet import se_resnet50
clf = se_resnet50(num_classes = 1000, pretrained = True)

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
