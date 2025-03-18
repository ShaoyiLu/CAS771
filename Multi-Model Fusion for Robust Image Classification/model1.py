import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

label_maps = {
    "A": {34: 0, 137: 1, 159: 2, 173: 3, 201: 4},
    "B": {34: 0, 135: 1, 202: 2, 80: 3, 24: 4},
    "C": {124: 0, 125: 1, 130: 2, 173: 3, 202: 4}
}

def load_pth(file_path, model_type):
    raw_data = torch.load(file_path, map_location="cpu")
    data = raw_data['data'].numpy()
    labels = raw_data['labels'].numpy()

    if not set(labels).issubset(set(label_maps[model_type].keys())):
        raise ValueError(f"Unexpected labels in {file_path}: {set(labels) - set(label_maps[model_type].keys())}")

    mapped_labels = np.array([label_maps[model_type][label] for label in labels])
    return data, mapped_labels

class CustomDataset(data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(8 * 8 * 128, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, val_loader, model_save_path, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = test_model(model, val_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with accuracy: {best_acc:.4f}")

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {total_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

def test_model(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return correct / total

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(VGGNet())}")

# Training three models
for i, (model_type, train_file, val_file) in enumerate(zip(
    ["A", "B", "C"],
    ["train_dataB_model_1.pth", "train_dataB_model_2.pth", "train_dataB_model_3.pth"],
    ["val_dataB_model_1.pth", "val_dataB_model_2.pth", "val_dataB_model_3.pth"]
)):
    print(f"\n==== Training Model {model_type} ====")

    train_images, train_labels = load_pth(train_file, model_type)
    val_images, val_labels = load_pth(val_file, model_type)

    train_dataset = CustomDataset(train_images, train_labels, transform)
    val_dataset = CustomDataset(val_images, val_labels, transform)

    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = VGGNet()

    model_save_path = f"Task {model_type} Model {i + 1} Trained.pth"

    train_model(model, train_loader, val_loader, model_save_path, epochs=100, lr=0.0001)

    model.load_state_dict(torch.load(model_save_path))
    val_acc = test_model(model, val_loader)
    print(f"Loaded model {model_type} accuracy: {val_acc:.4f}")

print("Training completed")
