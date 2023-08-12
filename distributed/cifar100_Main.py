import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import ResNet

# process data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load CIFAR_100
batch_size = 64
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
# init the model
num_classes = 100
resnet34 = ResNet.ResNet34(num_classes=num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet34.to(device)

# define optimizer and loss function
optimizer_front = optim.SGD([
    {'params': resnet34.conv1.parameters()},
    {'params': resnet34.bn1.parameters()},
    {'params': resnet34.layer1.parameters()},
    {'params': resnet34.layer2.parameters()},
    {'params': resnet34.fc1.parameters()}
], lr=0.001, momentum=0.9)  # update first two layer

optimizer_back = optim.SGD([
    {'params': resnet34.layer3.parameters()},
    {'params': resnet34.layer4.parameters()},
    {'params': resnet34.fc2.parameters()}
], lr=0.001, momentum=0.9)  # update layer3 and 4

criterion = torch.nn.CrossEntropyLoss()

train_losses = []
# train
num_epochs = 200
for epoch in range(num_epochs):
    resnet34.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_front.zero_grad()
        optimizer_back.zero_grad()

        outputs, extra_output = resnet34(inputs)
        loss_front = criterion(extra_output, labels)
        loss_front.backward(retain_graph=True)

        loss_back = criterion(outputs, labels)
        loss_back.backward()

        optimizer_front.step()
        optimizer_back.step()

        running_loss += loss_back.item()

    avg_train_loss = running_loss / len(trainloader)

    train_losses.append(avg_train_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}")

print("Training finished!")

# test
resnet34.eval()
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs, _ = resnet34(inputs)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
print(f"Test Accuracy: {accuracy:.2f}")

# plot learning curve
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
