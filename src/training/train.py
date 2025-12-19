# def train_one_epoch(model, loader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0

#     for images, labels in loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     return running_loss / len(loader)

#UPDATED VERSION
import torch

def train_one_epoch(model, loader, criterion, optimizer, device, writer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ✅ accuracy calculation
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(loader)
    train_acc = correct / total

    # TensorBoard logging
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_acc, epoch)

    return train_loss, train_acc
