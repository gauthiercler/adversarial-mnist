import torch
import torchvision
from torch import nn, optim
from torchvision.transforms import transforms

from classifier import ConvNet

PRINT_STEP = 1000


class ToCuda:
    def __call__(self, tensor):
        return tensor.cuda()


def train(net, loader, epochs, criterion, optimizer):

    correct = 0

    for epoch in range(epochs):
        for idx, data in enumerate(loader):

            X, y = data

            optimizer.zero_grad()

            outputs = net(X)

            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            correct += (predicted == y).sum().item()
            loss += loss.item()

            if idx % PRINT_STEP == PRINT_STEP - 1:
                print(f'epoch [{epoch + 1} {idx + 1}], loss {loss}, accuracy {correct / (loader.batch_size * PRINT_STEP) * 100:.3f}')
                correct = 0


def main():

    transformers = [transforms.ToTensor()]

    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        transformers.append(ToCuda())

    transform = transforms.Compose(transformers)

    trainset = torchvision.datasets.MNIST('.', download=True,
                                          train=True,
                                          transform=transform)

    loader = torch.utils.data.DataLoader(trainset,
                                         batch_size=6)

    net = ConvNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    train(net, loader, 10, criterion, optimizer)
    torch.save(net.state_dict(), 'model/trained.pth')

if __name__ == '__main__':
    main()