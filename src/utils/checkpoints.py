import os
import torch

def save_best_model(model, val_acc, best_val_acc, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)

    if val_acc > best_val_acc:
        path = os.path.join(save_dir, filename)
        torch.save(model.state_dict(), path)
        print("✅ Best model saved:", path)
        return val_acc

    return best_val_acc
