import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import ResNet

# process data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.expand(3, -1, -1))  # Expand to 3 channels
])

# load MNIST
batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# init the model
num_classes = 10
resnet34 = ResNet.ResNet34(num_classes=num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet34.to(device)

# define optimizer and loss function
optimizer = optim.SGD(resnet34.parameters(), lr=0.001, momentum=0.9)  # update first two layer

criterion = torch.nn.CrossEntropyLoss()

train_losses = []
test_losses = []
# train
num_epochs = 20
for epoch in range(num_epochs):
    resnet34.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet34(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_train_loss = running_loss / len(trainloader)

    train_losses.append(avg_train_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}")

    # test in every epoch
    resnet34.eval()
    test_running_loss = 0.0
    with torch.no_grad():
        for test_inputs, test_labels in testloader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = resnet34(test_inputs)
            test_loss = criterion(test_outputs, test_labels)
            test_running_loss += test_loss.item()

    avg_test_loss = test_running_loss / len(testloader)
    test_losses.append(avg_test_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")

print("Training finished!")

# test
resnet34.eval()

all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = resnet34(inputs)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
print(f"Test Accuracy: {accuracy:.2f}")

# plot the learning loss
'''
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
