'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import ResNet_8out
import time

device_id = 0
device = torch.device('cuda:{}'.format(device_id)) if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
print(device)
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
# load resnet101
model_name = 'resnet101_cifar10_8out'
print(model_name)
net = ResNet_8out.ResNet101(num_classes=10)
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

optimizer_5 = optim.SGD([
    {'params': net.layer5.parameters()},
    {'params': net.fc5.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update first two layer

optimizer_6 = optim.SGD([
    {'params': net.layer6.parameters()},
    {'params': net.fc6.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update layer3 and 4

optimizer_7 = optim.SGD([
    {'params': net.layer7.parameters()},
    {'params': net.fc7.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update layer3 and 4

optimizer_8 = optim.SGD([
    {'params': net.layer8.parameters()},
    {'params': net.fc8.parameters()}
], lr=0.1, momentum=0.9, weight_decay=5e-4)  # update layer3 and 4

scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=200)
scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=200)
scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_3, T_max=200)
scheduler_4 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_4, T_max=200)
scheduler_5 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_5, T_max=200)
scheduler_6 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_6, T_max=200)
scheduler_7 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_7, T_max=200)
scheduler_8 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_8, T_max=200)

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

        outputs, extra_1, extra_2, extra_3, extra_4, extra_5, extra_6, extra_7 = net(inputs)
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

        loss_8 = criterion(outputs, targets)
        loss_8.backward()

        optimizer_1.step()
        optimizer_2.step()
        optimizer_3.step()
        optimizer_4.step()
        optimizer_5.step()
        optimizer_6.step()
        optimizer_7.step()
        optimizer_8.step()

        train_loss += loss_8.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / len(trainloader), 100. * correct / total, correct, total))
    acc = 100. * correct / total
    train_acc.append(acc)
    train_losses.append(train_loss / len(trainloader))

    with open(r'./outcome/'+model_name+'/train_loss.txt', 'a') as f1:
        f1.write('epoch'+str(epoch)+':{:.3f}'.format(train_loss / len(trainloader)))
        f1.write('\n')
        f1.close()

    with open(r'./outcome/'+model_name+'/train_acc.txt', 'a') as f2:
        f2.write('epoch'+str(epoch)+':{:.3f}%'.format(acc))
        f2.write('\n')
        f2.close()


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
            outputs, _, _, _, _, _, _, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss / len(testloader), 100. * correct / total, correct, total))
    acc = 100. * correct / total
    test_acc.append(acc)

    if acc>test_acc_best:
        test_acc_best = acc
        epoch_acc_best = epoch
        model_best = net.state_dict()

    if (epoch+1)%10==0:
        save_path = r'./model/'+model_name+'/epoch'+str(epoch_acc_best)+'.pth'
        torch.save(model_best, save_path)
        print("Trained weights saved to:", save_path)

    test_losses.append(test_loss / len(testloader))

    with open(r'./outcome/'+model_name+'/test_loss.txt', 'a') as f3:
        f3.write('epoch'+str(epoch)+':{:.3f}'.format(test_loss / len(testloader)))
        f3.write('\n')
        f3.close()

    with open(r'./outcome/'+model_name+'/test_acc.txt', 'a') as f4:
        f4.write('epoch'+str(epoch)+':{:.3f}%'.format(acc))
        f4.write('\n')
        f4.close()


time0 = time.time()
with open(r'./outcome/'+model_name+'/train_loss.txt', 'w') as f:
    f.write('train_loss')
    f.write('\n')
    f.close()
with open(r'./outcome/'+model_name+'/train_acc.txt', 'w') as f:
    f.write('train_acc')
    f.write('\n')
    f.close()
with open(r'./outcome/'+model_name+'/test_loss.txt', 'w') as f:
    f.write('test_loss')
    f.write('\n')
    f.close()
with open(r'./outcome/'+model_name+'/test_acc.txt', 'w') as f:
    f.write('test_acc')
    f.write('\n')
    f.close()

for epoch in range(start_epoch, start_epoch + epoch_num):
    time0 = time.time()
    train(epoch)
    test(epoch)
    time1 = time.time()
    dt = time1 - time0
    h = dt//3600
    m = (dt - h*3600)//60
    s = dt - h*3600 - m*60
    print('epoch {} : {}h {}min {}s'.format(epoch, h, m, s))
    scheduler_1.step()
    scheduler_2.step()
    scheduler_3.step()
    scheduler_4.step()
    scheduler_5.step()
    scheduler_6.step()
    scheduler_7.step()
    scheduler_8.step()


# Save the trained weights
save_path = 'resnet101_dis8_cifar10.pth'
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
