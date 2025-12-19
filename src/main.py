# import torch
# from torch.utils.data import DataLoader

# from src.data.dataset import PalmLeafDataset
# from src.data.transforms import get_train_transforms, get_val_transforms
# from src.models.convnext import build_convnext
# from src.training.train import train_one_epoch
# from src.training.validate import validate
# from src.training.losses import get_loss
# from src.utils.checkpoints import save_best_model

# def main():
#     # ===== Config =====
#     num_classes = 9
#     batch_size = 4
#     epochs = 5
#     lr = 1e-3
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # ===== Data =====
#     train_dataset = PalmLeafDataset(
#         root_dir="Dataset/train",
#         transform=get_train_transforms()
#     )

#     val_dataset = PalmLeafDataset(
#         root_dir="Dataset/val",
#         transform=get_val_transforms()
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False
#     )

#     # ===== Model =====
#     model = build_convnext(
#         num_classes=num_classes,
#         pretrained=True,
#         freeze_backbone=True
#     ).to(device)

#     criterion = get_loss()
#     optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=lr
#     )

#     # ===== Training =====
#     best_val_acc = 0.0

#     for epoch in range(epochs):
#         train_loss = train_one_epoch(
#             model, train_loader, criterion, optimizer, device
#         )

#         val_loss, val_acc = validate(
#             model, val_loader, criterion, device
#         )

#         print(
#             f"Epoch [{epoch+1}/{epochs}] | "
#             f"Train Loss: {train_loss:.4f} | "
#             f"Val Loss: {val_loss:.4f} | "
#             f"Val Acc: {val_acc*100:.2f}%"
#         )

#         best_val_acc = save_best_model(
#             model,
#             val_acc,
#             best_val_acc,
#             save_dir="checkpoints",
#             filename="best_model.pth"
#         )

#     print("🎉 Training finished!")

# if __name__ == "__main__":
#     main()
#====================================================================================
#UPDATED VERSION
# import os
import yaml
import torch
from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report, accuracy_score

from src.utils.config import load_config
from src.utils.logger import get_writer
from src.utils.checkpoints import save_best_model
from src.data.dataset import PalmLeafDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.convnext import build_convnext
from src.training.train import train_one_epoch
from src.training.validate import validate
from src.training.losses import get_loss
from src.training.scheduler import get_scheduler
    

def main():
    config = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    writer = get_writer(
        config["logging"]["log_dir"],
        config["experiment"]["name"]
    )

    train_dataset = PalmLeafDataset(
        config["data"]["train_dir"],
        get_train_transforms(config["data"]["image_size"])
    )
    val_dataset = PalmLeafDataset(
        config["data"]["val_dir"],
        get_val_transforms(config["data"]["image_size"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    model = build_convnext(
        num_classes=config["data"]["num_classes"],
        pretrained=True,
        freeze_backbone=config["training"]["freeze_backbone"]
    ).to(device)

    criterion = get_loss(config["loss"]["label_smoothing"])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay = config["training"]["weight_decay"]
    )
    scheduler = get_scheduler(optimizer, config)


    best_val_acc = 0.0

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device, writer, epoch
        )

        # ✅ scheduler steps ONCE per epoch
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{config['training']['epochs']}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}%"
        )

        best_val_acc = save_best_model(
            model,
            val_acc,
            best_val_acc,
            config["logging"]["checkpoint_dir"],
            "best_model.pth"
        )

    writer.close()


if __name__ == "__main__":
    main()
