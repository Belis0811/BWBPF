'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import ResNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0

# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)
# load resnet50/101/152
net = ResNet.ResNet50()
# net = ResNet.ResNet101()
# net = ResNet.ResNet152()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
# define optimizer and loss function
optimizer_1 = optim.SGD([
    {'params': net.conv1.parameters()},
    {'params': net.bn1.parameters()},
    {'params': net.layer1.parameters()},
    {'params': net.fc1.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update first two layer

optimizer_2 = optim.SGD([
    {'params': net.layer2.parameters()},
    {'params': net.fc2.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update layer3 and 4

optimizer_3 = optim.SGD([
    {'params': net.layer3.parameters()},
    {'params': net.fc3.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update layer3 and 4

optimizer_4 = optim.SGD([
    {'params': net.layer4.parameters()},
    {'params': net.fc4.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update layer3 and 4
scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=400)
scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=400)
scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_3, T_max=400)
scheduler_4 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_4, T_max=400)

train_losses = []
test_losses = []


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        optimizer_3.zero_grad()
        optimizer_4.zero_grad()

        outputs, extra_1, extra_2, extra_3 = net(inputs)
        loss_1 = criterion(extra_1, targets)
        loss_1.backward(retain_graph=True)

        loss_2 = criterion(extra_2, targets)
        loss_2.backward(retain_graph=True)

        loss_3 = criterion(extra_3, targets)
        loss_3.backward(retain_graph=True)

        loss_4 = criterion(outputs, targets)
        loss_4.backward()

        optimizer_1.step()
        optimizer_2.step()
        optimizer_3.step()
        optimizer_4.step()

        train_loss += loss_4.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / len(trainloader), 100. * correct / total, correct, total))
    train_losses.append(train_loss / len(trainloader))


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss / len(testloader), 100. * correct / total, correct, total))
    test_losses.append(test_loss / len(testloader))


for epoch in range(start_epoch, start_epoch + 400):
    train(epoch)
    test(epoch)
    scheduler_1.step()
    scheduler_2.step()
    scheduler_3.step()
    scheduler_4.step()

# Save the trained weights
save_path = 'resnet50_dis4_cifar10.pth'
torch.save(net.state_dict(), save_path)
print("Trained weights saved to:", save_path)

# plot training loss
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# plot test loss
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
