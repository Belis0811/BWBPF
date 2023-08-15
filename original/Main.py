import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torchvision.models.resnet
import torch.nn as nn

import ResNet

# process data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load CIFAR_10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# init the model
num_classes = 10
resnet50 = ResNet.ResNet50()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_weight_path = "../resnet50-pre.pth"
missing_keys, unexpected_keys = resnet50.load_state_dict(torch.load(model_weight_path), strict=False)
inchannel = resnet50.fc.in_features
resnet50.fc = nn.Linear(inchannel, num_classes)
resnet50.to(device)

weight_decay = 0.0001

# define optimizer and loss function
optimizer = optim.SGD([
    {'params': resnet50.conv1.parameters()},
    {'params': resnet50.bn1.parameters()},
    {'params': resnet50.layer1.parameters()},
    {'params': resnet50.layer2.parameters()},
    {'params': resnet50.layer3.parameters()},
    {'params': resnet50.layer4.parameters()},
    {'params': resnet50.fc.parameters()}
], lr=0.001, momentum=0.9, weight_decay=weight_decay)  # update first two layer

criterion = torch.nn.CrossEntropyLoss()

train_losses = []
# test_losses = []
# train
num_epochs = 200
for epoch in range(num_epochs):
    resnet50.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_train_loss = running_loss / len(trainloader)
    train_losses.append(avg_train_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

    # test in every epoch
    '''
    resnet50.eval()
    test_running_loss = 0.0
    with torch.no_grad():
        for test_inputs, test_labels in testloader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = resnet50(test_inputs)
            test_loss = criterion(test_outputs, test_labels)
            test_running_loss += test_loss.item()

    avg_test_loss = test_running_loss / len(testloader)
    test_losses.append(avg_test_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")
    '''

print("Training finished!")

# test
resnet50.eval()

all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = resnet50(inputs)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
print(f"Test Loss Rate: {(1 - accuracy) * 100:.2f} %")

# plot the learning loss
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

'''
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
'''
