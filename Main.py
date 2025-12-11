import torch
from TinyViT import TinyViT
from Runner import Runner
import json

def main():
    # Load config
    # with open('Config_Cifar10.json', 'r') as f:
    with open('Config_Food101.json', 'r') as f:
        config = json.load(f)
        torch.manual_seed(config["experiment"]["seed"])

    # Define the model
    model = TinyViT(config)

    # Define the runner
    runner = Runner()
    runner.initialize(model, config)

    best_val = 0.0
    for epoch in range(1, config["training"]["epochs"] + 1):
        loss, acc, dur = runner.train_one_epoch(epoch)
        val_loss, val_acc = runner.evaluate(epoch)
        print(f"Epoch {epoch} finished in {dur:.1f}s | train_loss={loss:.4f} train_acc={acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # simple checkpoint
        if val_acc > best_val and epoch % config["training"]["save_every_k"] == 0:
            best_val = val_acc
            torch.save(model.state_dict(), "tiny_vit_ckpt.pth")
            print("Saved best model")

    print("Training complete. Best val acc:", best_val)

    # Quick attention extraction example on a single batch
    # model.eval()
    # xb, yb = next(iter(test_loader))
    # xb = xb.to(device)[:8]
    # with torch.no_grad():
    #     logits, attn_maps = model(xb, return_attn=True)
    # print("Extracted attention from", len(attn_maps), "layers. Each has shape", attn_maps[0].shape)
    # Save the model final
    torch.save(model.state_dict(), "tiny_vit_final.pth")
    print("Saved final model")

if __name__ == "__main__":
    main()