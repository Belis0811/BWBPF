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
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
# load resnet50
net = ResNet.ResNet50()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# define optimizer and loss function
optimizer_front = optim.SGD([
    {'params': net.conv1.parameters()},
    {'params': net.bn1.parameters()},
    {'params': net.layer1.parameters()},
    {'params': net.layer2.parameters()},
    {'params': net.fc1.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update first two layer

optimizer_back = optim.SGD([
    {'params': net.layer3.parameters()},
    {'params': net.layer4.parameters()},
    {'params': net.fc2.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update layer3 and 4
scheduler_front = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_front, T_max=200)
scheduler_back = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_back, T_max=200)

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

        optimizer_front.zero_grad()
        optimizer_back.zero_grad()

        outputs,extra_outputs = net(inputs)
        loss_front = criterion(extra_outputs, targets)
        loss_front.backward(retain_graph=True)

        loss_back = criterion(outputs, targets)
        loss_back.backward()

        optimizer_front.step()
        optimizer_back.step()

        train_loss += loss_back.item()
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
            outputs,_ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss / len(testloader), 100. * correct / total, correct, total))
    test_losses.append(test_loss / len(testloader))


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
    scheduler_front.step()
    scheduler_back.step()

# Save the trained weights
save_path = 'resnet50_dis_cifar10.pth'
torch.save(net.state_dict(), save_path)
print("Trained weights saved to:", save_path)

# plot training loss
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# plot test loss
plt.plot(train_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
