import argparse
import os
import torch
from torch.utils.data import DataLoader
from models.models import VanillaCNN, ResNet18Model, MobileNetV2Model
from datasets.image_dataset import ImageDataset
from cvtorchvision import cvtransforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from torchinfo import summary as torchinfo_summary
from tqdm import tqdm


def train(model, device, train_loader, optimizer, criterion, training_losses, training_acc):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model to train.
        device: Device to run the training on (CPU/GPU).
        train_loader: DataLoader for training data.
        optimizer: Optimizer for model training.
        criterion: Loss function.
        training_losses: List to store training loss values.
        training_acc: List to store training accuracy values.

    Returns:
        tuple: Last batch loss and accuracy for the epoch.
    """
    model.train()
    pbar = tqdm(train_loader, desc="Training")
    correct, processed = 0, 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        logits, softmax_out = model(data)
        _, target = torch.max(target.data, 1)  # Convert one-hot labels to class indices

        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        training_losses.append(loss.item())
        correct += (softmax_out.argmax(dim=1) == target).sum().item()
        processed += len(data)

        pbar.set_description(
            f"Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:.2f}%"
        )
        training_acc.append(100 * correct / processed)

    return loss.item(), 100 * correct / processed


def validate(model, device, valid_loader, criterion, valid_losses, test_acc):
    """
    Validate the model on the validation dataset.

    Args:
        model: PyTorch model to validate.
        device: Device to run the validation on (CPU/GPU).
        valid_loader: DataLoader for validation data.
        criterion: Loss function.
        valid_losses: List to store validation loss values.
        test_acc: List to store validation accuracy values.

    Returns:
        tuple: Average validation loss and accuracy.
    """
    model.eval()
    valid_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)

            logits, softmax_out = model(data)
            _, target = torch.max(target.data, 1)  # Convert one-hot labels to class indices

            valid_loss += criterion(logits, target).item()
            correct += (softmax_out.argmax(dim=1) == target).sum().item()

    valid_loss /= len(valid_loader.dataset)
    valid_losses.append(valid_loss)
    accuracy = 100.0 * correct / len(valid_loader.dataset)
    test_acc.append(accuracy)

    print(f"Validation: Avg loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return valid_loss, accuracy


def main(args):
    """
    Main function to set up and run training and validation.

    Args:
        args: Parsed command-line arguments.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    print("Loading dataset...")
    train_transform = cvtransforms.Compose([
        cvtransforms.Resize((224, 224)),
        cvtransforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.RandomRotation(10),
        cvtransforms.ToTensor()
    ])
    valid_transform = cvtransforms.Compose([
        cvtransforms.Resize((224, 224)),
        cvtransforms.ToTensor()
    ])

    train_data = ImageDataset(os.path.join(args.data_path, "train"), transform=train_transform)
    valid_data = ImageDataset(os.path.join(args.data_path, "test"), transform=valid_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size_train, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size_valid, shuffle=False, pin_memory=True)

    print(f"Train samples: {len(train_data)}, Validation samples: {len(valid_data)}")

    # Model selection
    print(f"Initializing model: {args.backbone}")
    model_map = {
        "vanilla_cnn": VanillaCNN,
        "resnet18": ResNet18Model,
        "mobilenet_v2": MobileNetV2Model
    }

    if args.backbone not in model_map:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    model = model_map[args.backbone](num_classes=args.num_classes).to(device)
    summary(model, input_size=(3, 224, 224))
    torchinfo_summary(model, (3, 224, 224), batch_dim=0, col_names=("input_size", "output_size", "num_params"))

    # Loss, optimizer, and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    # Training and validation
    best_valid_acc = 0
    training_losses, valid_losses = [], []
    training_acc, test_acc = [], []

    for epoch in range(1, args.epochs + 1):
        print(f"EPOCH {epoch}/{args.epochs}")
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, training_losses, training_acc)
        scheduler.step()
        valid_loss, valid_acc = validate(model, device, valid_loader, criterion, valid_losses, test_acc)

        if valid_acc > best_valid_acc:
            torch.save(model.state_dict(), f"{args.saved_model_dir}/best_model.pth")
            best_valid_acc = valid_acc

    torch.save(model.state_dict(), f"{args.saved_model_dir}/best_model.pth")
    print(f"Training complete. Best validation accuracy: {best_valid_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for chest X-ray classification.")
    parser.add_argument("--data_path", type=str, default="../chest_xray/", help="Path to dataset.")
    parser.add_argument("--batch_size_train", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--batch_size_valid", type=int, default=8, help="Batch size for validation.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes.")
    parser.add_argument("--backbone", type=str, choices=["vanilla_cnn", "resnet18", "mobilenet_v2"], default="resnet18", help="Model backbone.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--saved_model_dir", type=str, default="./weights/resnet18/", help="Path to save the best model.")
    args = parser.parse_args()
    main(args)
