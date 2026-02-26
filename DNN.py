import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from SNN import SNN

# 56.5% trained with 58% vanilla SNN
# 44% trained with 58% ResNet18 distilled SNN, not very good at all

class DNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DNN, self).__init__()

        input_dim = 3 * 32 * 32
        hidden_dim1 = 2048
        hidden_dim2 = 512
        hidden_dim3 = 64

        self.net = nn.Sequential(
            nn.Flatten(),                   
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, num_classes), 
        )
        self.softmax = nn.Softmax(dim=1)

    def forward_logits(self, x):
        return self.net(x)

    def forward(self, x):
        logits = self.forward_logits(x)
        return self.softmax(logits)


def train_with_teacher(model, device, train_loader, optimizer, criterion, epoch, teacher):
    model.train()
    running_loss = 0.0
    teacher.eval()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model.forward_logits(data)
         
        with torch.no_grad():
            target = teacher.forward_logits(data)

        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )

    return running_loss / len(train_loader)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model.forward_logits(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return test_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    model = DNN(num_classes=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-8,
    )
    criterion = nn.MSELoss()

    print('loading teacher...')
    teacher = SNN().to(device)
    teacher.load_state_dict(torch.load('models/resnet1_cifar10_teacher_resnet18.pth', map_location=device)) 

    for epoch in range(1, 201):
        train_loss = train_with_teacher(model, device, train_loader, optimizer, criterion, epoch, teacher)
        print(f"Epoch {epoch}: Train loss {train_loss:.6f}")
        val_loss, _ = test(model, device, test_loader)
        scheduler.step(val_loss)

    torch.save(model.state_dict(), "models/DNN_cifar10_teacher_distilled_SNN.pth")
    print("Model saved to models/DNN_cifar10_teacher_distilled_SNN.pth")


if __name__ == "__main__":
    main()
