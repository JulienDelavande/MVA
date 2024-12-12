import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchattacks
import argparse

# Load classifier function
def load_classifier(model_name):
    if model_name == "GFZ":
        model = torch.load("GFZ.pth")
        model.eval()
        return model
    elif model_name == "DFZ":
        model = torch.load("DFZ.pth")
        model.eval()
        return model
    else:
        raise ValueError("Invalid model name")

# Main function to test adversarial attacks
def test_attacks(data_name, model_name, attack_method, eps, batch_size=100, targeted=False, save=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    if data_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    elif data_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    else:
        raise ValueError("Invalid dataset name")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = load_classifier(model_name)
    model.to(device)

    # Select attack method
    if attack_method == 'fgsm':
        attack = torchattacks.FGSM(model, eps=eps)
    elif attack_method == 'pgd':
        attack = torchattacks.PGD(model, eps=eps, alpha=0.01, steps=40)
    elif attack_method == 'mim':
        attack = torchattacks.MIFGSM(model, eps=eps, decay=1.0)
    else:
        raise ValueError("Invalid attack method")

    total = 0
    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack(images, labels if targeted else None)
        outputs = model(adv_images)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    success_rate = 1 - (correct / total)
    print(f"Attack Success Rate: {success_rate * 100:.2f}%")

    # Optionally save adversarial examples
    if save:
        torch.save(adv_images.cpu(), "adv_examples.pt")
        print("Adversarial examples saved to adv_examples.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run adversarial attack experiments.')
    parser.add_argument('--batch_size', '-B', type=int, default=100)
    parser.add_argument('--data', '-D', type=str, default='mnist')
    parser.add_argument('--attack', '-A', type=str, default='pgd')
    parser.add_argument('--eps', '-e', type=float, default=0.1)
    parser.add_argument('--victim', '-V', type=str, default='example_model')
    parser.add_argument('--save', '-S', action='store_true', default=False)

    args = parser.parse_args()
    test_attacks(data_name=args.data,
                 model_name=args.victim,
                 attack_method=args.attack,
                 eps=args.eps,
                 batch_size=args.batch_size,
                 targeted=False,
                 save=args.save)
