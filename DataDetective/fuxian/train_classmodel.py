import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from fuxian.disassemble_dataset import DisassembledDataSet
from TruncatedLoss import TruncatedLoss


def build_dataset(mask_type,class_num):
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]
    )
    disassembled_dataset = DisassembledDataSet(
        img_root_dir, 
        annotation_path,
        class_num = class_num,
        mask_type = mask_type,
        transforms=data_transform)
    return disassembled_dataset

def build_ResNet50(class_num):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, class_num)
    return model

def build_criterion(is_LNL,train_dataset):
    if is_LNL:
        criterion = TruncatedLoss(trainset_size=len(train_dataset)).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion

def train_one_epoch(epoch,model,train_dataloader,
                    optimizer,criterion,
                    is_LNL,
                    model_save_dir,
                    device):
    print("epoch: %d, lr: %f" % (epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_sum = 0
    if (epoch + 1) >= 3 and (epoch + 1) % 3 == 0 and is_LNL:
        best_checkpoint = torch.load(os.path.join(model_save_dir,"best.pth"), map_location="cpu")
        model.load_state_dict(best_checkpoint["model"])
        model.eval()
        for batch_idx, (inputs, targets, indexes) in enumerate(train_dataloader):
            print("\rrunning update_weight:{} / {}".format(batch_idx, len(train_dataloader)), end="")
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            criterion.update_weight(outputs, targets, indexes)
        last_checkpoint = torch.load(os.path.join(model_save_dir,f"epoch_{epoch-1}.pth"), map_location="cpu")
        model.load_state_dict(last_checkpoint['model'])
        model.train()

    for i, (inputs, labels, indexes) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        # forward
        outputs = model(inputs)
        if is_LNL:
            loss = criterion(outputs, labels, indexes)
        else:
            loss = criterion(outputs, labels)
        loss_sum += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Training progress bar
        print("\rEpoch: {}/{} | Step: {}/{} | Loss: {:.4f}".format(epoch + 1, epoches, i + 1, len(train_dataloader),
                                                                    loss.item()), end="")
    loss_avg = round(loss_sum / len(train_dataloader),4)
    return loss_avg


def val_one_epoch(model,val_dataloader,device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels, indexes in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # test progress bar
            print("\rTest: {}/{}".format(total, len(val_dataloader)), end="")

    print("Accuracy of the val images: {} %".format(100 * correct / total))

    acc = 100 * correct / total
    return acc

def train():
    train_disassembled_dataset = build_dataset(mask_type,class_num)
    train_dataloader = DataLoader(train_disassembled_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(train_disassembled_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda:0")
    model = build_ResNet50(class_num)
    model.to(device)
    is_LNL = True
    criterion = build_criterion(is_LNL=is_LNL,train_dataset=train_disassembled_dataset)
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 11], gamma=0.1)

    best_acc = 0.0
    # train
    for epoch in range(epoches):
        loss_avg = train_one_epoch(epoch,model,train_dataloader,
                    optimizer,criterion,
                    is_LNL,
                    model_save_dir,
                    device)
        lr_scheduler.step()
        print(" | Loss_avg: {:.4}".format(loss_avg))
        
        val_acc = val_one_epoch(model,val_dataloader,device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss_avg,
                "acc": val_acc
            }, os.path.join(model_save_dir,"best.pt"))

        print("Now best acc: {} %".format(best_acc))
        # save checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss_avg,
            "acc": val_acc
        }, os.path.join(model_save_dir,f"epoch_{epoch}.pt"))


if __name__ == "__main__":
    img_root_dir = "/data/mml/data_debugging/datasets/VOC2012-coco/train", 
    annotation_path = "/data/mml/data_debugging/datasets/VOC2012-coco/train/_annotations.coco_error.json",
    mask_type = "other objects" # other objects|crop
    class_num = 12
    epoches = 13
    model_save_dir = f"/data/mml/data_debugging/saved_models/DataDetective/{mask_type}"
    train()


# resume
# checkpoint = torch.load('./models/resnet50_voc_epoch_10.pth', map_location="cpu")
# model.load_state_dict(checkpoint["model"])
# optimizer.load_state_dict(checkpoint["optimizer"])
# epoch = checkpoint["epoch"]
# loss = checkpoint["loss"]
# acc=checkpoint["acc"]
# print("checkpoint acc = ",acc)