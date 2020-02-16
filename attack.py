import argparse

import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms

from classifier import ConvNet

argparser = argparse.ArgumentParser()
argparser.add_argument('--test-size', type=int, default=100, help='size of the test size')
argparser.add_argument('--epsilon', type=float, default=0.1, help='epsilon value')

args = argparser.parse_args()


def evaluate(model, loader, epsilon):
    correct = 0
    criterion = nn.CrossEntropyLoss()

    for idx, data in enumerate(loader):
        X, y = data

        X.requires_grad = True
        outputs = model(X)

        _, predicted = torch.max(outputs.data, 1)
        if predicted == y:
            loss = criterion(outputs, y)
            model.zero_grad()

            loss.backward()

            data_grad = X.grad.data
            attack_image = fgsm(X, data_grad, e=epsilon)

            outputs = model(attack_image)
            _, predicted_ = torch.max(outputs.data, 1)

            if predicted == predicted_:
                correct += 1
    print(correct / len(loader))


def fgsm(input_image, grad, e=0.05):
    new_image = e * grad.sign() + input_image
    return torch.clamp(new_image, 0, 1)


def main():
    model = ConvNet()
    model.load_state_dict(torch.load('model/trained.pth'))
    print(model)

    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST('.', download=True,
                                         train=False,
                                         transform=transform)
    subset = torch.utils.data.Subset(testset, indices=range(args.test_size))
    loader = torch.utils.data.DataLoader(subset,
                                         batch_size=1)
    evaluate(model, loader, args.epsilon)


if __name__ == '__main__':
    main()
