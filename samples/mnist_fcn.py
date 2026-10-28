import torch
import torch.nn as nn
from softhebb_model.hebb import SoftHebbLinear
from softhebb_model.residual import Residual
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
  # constants
  stacks = 4
  batch_size = 32
  epochs = 10
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
  ])

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # dataset
  train_dataset, validation_dataset = torch.utils.data.random_split(
    datasets.MNIST(root='./data', train=True, download=True, transform=transform),
    [50000, 10000]
  )
  test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

  # batch loader
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  validation_loader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
  
  # model def
  def layer():
    '''
    Linear Softhebb layer with Pre-Norm.
    For experimenting deep softhebb network.
    '''

    return Residual(
      nn.Sequential(
        nn.LayerNorm((28 * 28,)),
        SoftHebbLinear(28 * 28, 28 * 28),
      ),
    )
  model = nn.Sequential(
    nn.Sequential(*[ layer() for _ in range(stacks) ]),
    nn.Linear(28 * 28, 10),
  ).to(device);

  # train
  for epoch in range(epochs):
    model.train()
    train_loss, train_acc = train(device, train_loader, model)

    model.eval()
    validation_acc = test(device, validation_loader, model)
    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Validation Acc: {validation_acc:.2f}%")

  # test
  model.eval()
  test_acc = test(device, test_loader, model)
  print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
  
def test(
  device: torch.device,
  test_loader: DataLoader,
  model: nn.Module,
):
  test_correct, test_total = 0, 0
  with torch.inference_mode():
    for x, y in test_loader:
      x, y = x.to(device), y.to(device)
      outputs = model(x)
      predicted = outputs.argmax(1)
      test_total += y.size(0)
      test_correct += predicted.eq(y).sum().item()

  test_acc = 100. * test_correct / test_total
  return test_acc

def train(
  device: torch.device,
  train_loader: DataLoader,
  model: nn.Module,
):
  total_loss, correct, total = 0, 0, 0

  classifier_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()
  for x, y in train_loader:
    x, y = x.to(device), y.to(device)

    outputs = model(x)
    loss = criterion(outputs, y)
                  
    classifier_optimizer.zero_grad()
    loss.backward()
    classifier_optimizer.step()

    total_loss += loss.item() * x.size(0)
    predicted = outputs.argmax(1)
    total += y.size(0)
    correct += predicted.eq(y).sum().item()
                  
  train_loss = total_loss / total
  train_acc = 100. * correct / total
  return (train_loss, train_acc)

if __name__ == "__main__":
  main()