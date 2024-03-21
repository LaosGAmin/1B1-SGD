import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from LBLSGD import LBLSGD, LBL_SGD_Function as LBLFunction

class MNIST_Net(nn.Module):
    def __init__(self, lr=0.1):
        super(MNIST_Net, self).__init__()
        LBLFunction.set_lr(lr)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # 28*28*32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output_size = 14*14*32
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)    # 12*12*64
        self.relu2 = torch.nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output_size = 6*6*64
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0)   # 4*4*128
        self.relu3 = torch.nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2*2*128
        self.Flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear7 = nn.Linear(2*2*128, 256, bias=True)
        self.relu7 = torch.nn.ReLU()
        self.linear8 = nn.Linear(256, 128, bias=True)
        self.relu8 = torch.nn.ReLU()
        self.linear9 = nn.Linear(128, 10, bias=True)
        self.blocks = nn.ModuleList([self.conv1, self.relu1, self.pool1,
                                     self.conv2, self.relu2, self.pool2,
                                     self.conv3, self.relu3, self.pool3,
                                     self.Flatten,
                                     self.linear7, self.relu7, self.linear8, self.relu8, self.linear9])

    def forward(self, x, save_memory=True):
        if save_memory:
            LBLFunction.set_data(x)
            x = LBLSGD(None, self.blocks[0:3], x)
            x = LBLSGD(self.blocks[0:3], self.blocks[3:6], x)
            x = LBLSGD(self.blocks[0:6], self.blocks[6:10], x)
            x = LBLSGD(self.blocks[0:10], self.blocks[10:12], x)
            x = LBLSGD(self.blocks[0:12], self.blocks[12:14], x)
            x = LBLSGD(self.blocks[0:14], self.blocks[14:], x)
            return x
        else:
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.pool3(x)
            x = x.view(x.size(0), -1)
            # x = self.Flatten(x)

            x = self.linear7(x)
            x = self.relu7(x)
            x = self.linear8(x)
            x = self.relu8(x)
            x = self.linear9(x)
            return x


def RGB2GRAY(img):
    Img = img.convert('L')  # 256-color
    threshold = 10
    table = []
    for i in range(256):
        if i < threshold:  # If the pixel is less than 10, it is black. (black is 0)
            table.append(0)
        else:
            table.append(1)

    photo = Img.point(table, '1')
    return photo


if __name__ == '__main__':

    EPOCH = 40
    BATCH_SIZE = 200
    LR = 0.1

    trans = transforms.Compose([
        transforms.Lambda(lambda the_img: RGB2GRAY(the_img)),  # Convert to binary images of channel = 1
        transforms.ToTensor(),
    ])
    train_dataset = ImageFolder(root='./mnist/MNIST_image/raw/train', transform=trans)
    test_dataset = ImageFolder(root='./mnist/MNIST_image/raw/test', transform=trans)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=2000, shuffle=True, drop_last=True)

    cnn = MNIST_Net()
    print(cnn)  # net architecture
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step, (data, target) in enumerate(train_dataloader):
            target = torch.unsqueeze(target, 0).transpose(0, 1)
            zero = torch.zeros(BATCH_SIZE, 10)
            target = zero.scatter_(1, target, 1).clone()
            target.requires_grad = True
            data.requires_grad = True
            output = cnn(data)
            loss = loss_func(output, target)
            print("loss:", loss)
            loss.backward()

