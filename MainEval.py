import torch
from TinyViT import TinyViT
from Runner import Runner
import json

def main():
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
        config["experiment"]["log_per_class_acc"] = True
        config["training"] = {"batch_size": 128}

    torch.manual_seed(config["experiment"]["seed"])

    # Define the model
    model = TinyViT(config)
    model.load_state_dict(torch.load('tiny_vit_final.pth'))
    model.eval()

    # Define the runner
    runner = Runner()
    runner.initialize(model, config)

    class_names = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]

    val_loss, val_acc = runner.evaluate(0, class_names, 5)
    print(f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

if __name__ == "__main__":
    main()