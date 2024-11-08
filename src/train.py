import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from model import mnet

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST("../data/train", True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
print(len(train_data))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = mnet().to(device)

loss = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

st = time.time()
epoch = 10
for i in range(epoch):
    print(i, len(train_loader))
    model.train()
    temp = 0
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        loss_v = loss(model(img), label)
        temp = loss_v.item()
        optimizer.zero_grad()
        loss_v.backward()
        optimizer.step()
    print(temp)
    if temp < 0.001:
        break

print(time.time() - st)
torch.save(model, "../model3.pth")