'''Train Tiny ImageNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import ResNet_12out
import tiny_imagenet_loader

device_id = 0
device = torch.device('cuda:{}'.format(device_id)) if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# get directory
train_directory = '../tiny-imagenet-200/train'
test_directory = '../tiny-imagenet-200/'
# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.ImageFolder(
    root=train_directory, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=0)

testset = tiny_imagenet_loader.TinyImageNet_load(test_directory, train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=0)

# load resnet101
net = ResNet_12out.ResNet101(num_classes=200)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# define optimizer and loss function
optimizer_1 = optim.SGD([
    {'params': net.conv1.parameters()},
    {'params': net.bn1.parameters()},
    {'params': net.layer1.parameters()},
    {'params': net.fc1.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update first block

optimizer_2 = optim.SGD([
    {'params': net.layer2.parameters()},
    {'params': net.fc2.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update block2

optimizer_3 = optim.SGD([
    {'params': net.layer3.parameters()},
    {'params': net.fc3.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update block3

optimizer_4 = optim.SGD([
    {'params': net.layer4.parameters()},
    {'params': net.fc4.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update block4

optimizer_5 = optim.SGD([
    {'params': net.layer5.parameters()},
    {'params': net.fc5.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update block5

optimizer_6 = optim.SGD([
    {'params': net.layer6.parameters()},
    {'params': net.fc6.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update block6

optimizer_7 = optim.SGD([
    {'params': net.layer7.parameters()},
    {'params': net.fc7.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update block7

optimizer_8 = optim.SGD([
    {'params': net.layer8.parameters()},
    {'params': net.fc8.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update block8

optimizer_9 = optim.SGD([
    {'params': net.layer9.parameters()},
    {'params': net.fc9.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update block9

optimizer_10 = optim.SGD([
    {'params': net.layer10.parameters()},
    {'params': net.fc10.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update block10

optimizer_11 = optim.SGD([
    {'params': net.layer11.parameters()},
    {'params': net.fc11.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update block11

optimizer_12 = optim.SGD([
    {'params': net.layer12.parameters()},
    {'params': net.fc12.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update block12

scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=400)
scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=400)
scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_3, T_max=400)
scheduler_4 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_4, T_max=400)
scheduler_5 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_5, T_max=400)
scheduler_6 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_6, T_max=400)
scheduler_7 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_7, T_max=400)
scheduler_8 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_8, T_max=400)
scheduler_9 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_9, T_max=400)
scheduler_10 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_10, T_max=400)
scheduler_11 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_11, T_max=400)
scheduler_12 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_12, T_max=400)

train_losses = []
test_losses = []
train_acc = []
test_acc = []
test_acc_best = 0
epoch_acc_best = 0
model_best = None
epoch_num = 200


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
        optimizer_5.zero_grad()
        optimizer_6.zero_grad()
        optimizer_7.zero_grad()
        optimizer_8.zero_grad()
        optimizer_9.zero_grad()
        optimizer_10.zero_grad()
        optimizer_11.zero_grad()
        optimizer_12.zero_grad()

        outputs, extra_1, extra_2, extra_3, extra_4, extra_5, extra_6, extra_7, extra_8, extra_9, extra_10, extra_11 = net(
            inputs)
        loss_1 = criterion(extra_1, targets)
        loss_1.backward(retain_graph=True)

        loss_2 = criterion(extra_2, targets)
        loss_2.backward(retain_graph=True)

        loss_3 = criterion(extra_3, targets)
        loss_3.backward(retain_graph=True)

        loss_4 = criterion(extra_4, targets)
        loss_4.backward(retain_graph=True)

        loss_5 = criterion(extra_5, targets)
        loss_5.backward(retain_graph=True)

        loss_6 = criterion(extra_6, targets)
        loss_6.backward(retain_graph=True)

        loss_7 = criterion(extra_7, targets)
        loss_7.backward(retain_graph=True)

        loss_8 = criterion(extra_8, targets)
        loss_8.backward(retain_graph=True)

        loss_9 = criterion(extra_9, targets)
        loss_9.backward(retain_graph=True)

        loss_10 = criterion(extra_10, targets)
        loss_10.backward(retain_graph=True)

        loss_11 = criterion(extra_11, targets)
        loss_11.backward(retain_graph=True)

        loss_12 = criterion(outputs, targets)
        loss_12.backward()

        optimizer_1.step()
        optimizer_2.step()
        optimizer_3.step()
        optimizer_4.step()
        optimizer_5.step()
        optimizer_6.step()
        optimizer_7.step()
        optimizer_8.step()
        optimizer_9.step()
        optimizer_10.step()
        optimizer_11.step()
        optimizer_12.step()

        train_loss += loss_12.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / len(trainloader), 100. * correct / total, correct, total))
    acc = 100. * correct / total
    train_acc.append(acc)
    train_losses.append(train_loss / len(trainloader))


def test(epoch):
    global test_acc_best
    global epoch_acc_best
    global model_best
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _, _, _, _, _, _, _, _, _, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss / len(testloader), 100. * correct / total, correct, total))
    acc = 100. * correct / total
    test_acc.append(acc)

    if acc > test_acc_best:
        test_acc_best = acc
        epoch_acc_best = epoch
        model_best = net.state_dict()

    test_losses.append(test_loss / len(testloader))


for epoch in range(start_epoch, start_epoch + epoch_num):
    train(epoch)
    test(epoch)
    scheduler_1.step()
    scheduler_2.step()
    scheduler_3.step()
    scheduler_4.step()
    scheduler_5.step()
    scheduler_6.step()
    scheduler_7.step()
    scheduler_8.step()
    scheduler_9.step()
    scheduler_10.step()
    scheduler_11.step()
    scheduler_12.step()

# Save the trained weights
save_path = 'resnet101_dis12_tinyImagenet.pth'
torch.save(net.state_dict(), save_path)
print("Trained weights saved to:", save_path)
