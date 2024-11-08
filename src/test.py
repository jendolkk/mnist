import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])
test_data = datasets.MNIST("../data/test", False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

device = torch.device("cuda")
model = torch.load("../model2.pth")
model = model.to(device)
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算准确率
acc = 100 * correct / total
print(acc)