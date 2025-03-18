import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CAS771Dataset(Dataset):
    def __init__(self, data, labels, transform=False):
        self.data = data
        self.labels = labels
        self.transform = ToTensor() if transform else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform and not isinstance(img, torch.Tensor):
            img = self.transform(img)
        return img, label

def load_data(data_path):
    raw_data = torch.load(data_path)
    data = raw_data['data']
    labels = raw_data['labels']
    return data, labels

class AlexNet(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def train_model(model, train_loader, test_loader, num_epochs=20, lr=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        test_acc = test_model(model, test_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, marker='s', linestyle='-', color='green', label='Train Acc')
    plt.plot(range(1, num_epochs+1), test_accuracies, marker='d', linestyle='-', color='red', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training & Testing Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return model

def load_and_inspect_model(model_save_path):
    model = AlexNet(num_classes=10)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    print(model)

    last_layer = model.fc2
    num_classes = last_layer.out_features
    print("Number of classes in the model:", num_classes)

    return model

def train_single_model(data_path):
    data, labels = load_data(data_path)
    unique_labels = sorted(set(labels))
    num_classes_per_model = 5
    total_classes = len(unique_labels)
    models = []

    for i in range(0, total_classes, num_classes_per_model):
        subset_labels = unique_labels[i:i+num_classes_per_model]
        label_map = {label: idx for idx, label in enumerate(subset_labels)}
        subset_data = [data[j] for j in range(len(labels)) if labels[j] in subset_labels]
        subset_labels = [label_map[labels[j]] for j in range(len(labels)) if labels[j] in subset_labels]

        dataset = CAS771Dataset(subset_data, subset_labels, transform=True)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        model = AlexNet(num_classes=10)
        trained_model = train_model(model, train_loader, test_loader)
        models.append(trained_model)

        model_save_path = data_path.replace('.pth', '_trained.pth')
        torch.save(trained_model.state_dict(), model_save_path)
        print(f'Model trained and saved to {model_save_path}.')

        load_and_inspect_model(model_save_path)

    return models

# train_single_model('Model1/model1_train.pth')
train_single_model('Model2/model2_train.pth')
# train_single_model('Model3/model3_train.pth')
