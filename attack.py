import argparse

import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms

from classifier import ConvNet

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--test-size', type=int, default=1000, help='number of sample evaluated')
argparser.add_argument('--epsilon', type=float, default=0.05, help='epsilon value')
argparser.add_argument('--target-value', type=int, default=0, choices=range(10), help='Targeted class')
argparser.add_argument('--targeted', action='store_true', help='Enable class target on fgsm')

args = argparser.parse_args()


class AdversarialAttack():
    def __init__(self, model, loader, criterion, targeted, target_value):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.targeted = targeted
        self.target = torch.tensor([target_value])

    def fgsm(self, X, y, outputs, e):
        loss = self.criterion(outputs, self.target if self.targeted else y)
        if self.targeted:
            loss *= -1
        self.model.zero_grad()
        loss.backward()

        grad = X.grad.data
        X_adv = X + e * grad.sign()

        return torch.clamp(X_adv, 0, 1)

    def evaluate(self, epsilon):
        correct = 0

        for idx, data in enumerate(self.loader):
            X, y = data

            X.requires_grad = True
            # y.require_grad = True

            outputs = self.model(X)

            _, y_pred = torch.max(outputs.data, 1)

            if y_pred == y:
                attack_image = self.fgsm(X, y, outputs, epsilon)

                outputs = self.model(attack_image)
                _, y_pred_ = torch.max(outputs.data, 1)

                if y_pred == y_pred_:
                    correct += 1
        print(correct / len(self.loader))


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

    adv = AdversarialAttack(model, loader, criterion=nn.CrossEntropyLoss(),
                            targeted=args.targeted, target_value=args.target_value)

    adv.evaluate(args.epsilon)


if __name__ == '__main__':
    main()
