import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torchvision.models import ResNet50_Weights
import torchvision.models as models
import ResNet

# process data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load CIFAR_10
batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
# init the model
num_classes = 10
resnet50 = ResNet.ResNet50(num_classes=num_classes)

weights = ResNet50_Weights.DEFAULT
resnet50_pretrained = models.resnet50(weights=weights)
pretrained_state_dict = resnet50_pretrained.state_dict()

# Remove keys not present in custom ResNet state_dict
model_state_dict = resnet50.state_dict()
pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}

# Update model's state_dict
model_state_dict.update(pretrained_state_dict)
resnet50.load_state_dict(model_state_dict)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50.to(device)
weight_decay = 0.001

# define optimizer and loss function
optimizer_1 = optim.SGD([
    {'params': resnet50.conv1.parameters()},
    {'params': resnet50.bn1.parameters()},
    {'params': resnet50.layer1.parameters()},
    {'params': resnet50.fc1.parameters()}
], lr=0.001, momentum=0.9)  # update first two layer

optimizer_2 = optim.SGD([
    {'params': resnet50.layer2.parameters()},
    {'params': resnet50.fc2.parameters()}
], lr=0.001, momentum=0.9)  # update layer3 and 4

optimizer_3 = optim.SGD([
    {'params': resnet50.layer3.parameters()},
    {'params': resnet50.fc3.parameters()}
], lr=0.001, momentum=0.9)  # update layer3 and 4

optimizer_4 = optim.SGD([
    {'params': resnet50.layer4.parameters()},
    {'params': resnet50.fc4.parameters()}
], lr=0.001, momentum=0.9)  # update layer3 and 4

criterion = torch.nn.CrossEntropyLoss()

train_losses = []
# train
num_epochs = 150
for epoch in range(num_epochs):
    resnet50.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        optimizer_3.zero_grad()
        optimizer_4.zero_grad()

        outputs, extra_1, extra_2,extra_3 = resnet50(inputs)
        loss_1 = criterion(extra_1, labels)
        loss_1.backward(retain_graph=True)

        loss_2 = criterion(extra_2, labels)
        loss_2.backward(retain_graph=True)

        loss_3 = criterion(extra_3, labels)
        loss_3.backward(retain_graph=True)

        loss_4 = criterion(outputs, labels)
        loss_4.backward()

        optimizer_1.step()
        optimizer_2.step()
        optimizer_3.step()
        optimizer_4.step()

        running_loss += loss_4.item()

    avg_train_loss = running_loss / len(trainloader)

    train_losses.append(avg_train_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}")

print("Training finished!")

# test
resnet50.eval()
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs, _ = resnet50(inputs)
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
