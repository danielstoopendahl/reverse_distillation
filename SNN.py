import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 43% ResNet-2 as teacher
# 59% ResNet-6 as teacher
# 62% ResNet-18 as teacher

class SNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SNN, self).__init__()

        input_dim = 3 * 32 * 32
        hidden_dim = 16384

        self.net = nn.Sequential(
            nn.Flatten(),                   
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes), 
        )
        self.softmax = nn.Softmax(dim=1)

    def forward_logits(self, x):
        return self.net(x)

    def forward(self, x):
        logits = self.forward_logits(x)
        return self.softmax(logits)

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model.forward_logits(data)
            target_oh = F.one_hot(target, num_classes=10).float() 
            test_loss += criterion(logits, target_oh).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return test_loss, accuracy



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    model = SNN(num_classes=10).to(device)
    model.load_state_dict(torch.load('models/resnet1_cifar10.pth', map_location=device))

    criterion = nn.MSELoss()

    print('loading teacher...')
    teacher = SNN().to(device)
    teacher.load_state_dict(torch.load('models/resnet1_cifar10.pth', map_location=device)) 

    test(model, device, test_loader, criterion)


if __name__ == "__main__":
    main()
