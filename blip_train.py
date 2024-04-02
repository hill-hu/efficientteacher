import argparse
import os
import sys
from pathlib import Path

from utils.general import increment_path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # EfficientTeacher root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    opt = parser.parse_args()
    # print_args(vars(opt))
    return opt


from datasets import load_dataset
import torch


def main(opt):
    save_dir = '../runs_yolov5/blip'
    save_dir = increment_path(save_dir, mkdir=True)
    print("save_dir :", save_dir)
    processor = AutoProcessor.from_pretrained(r"blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        r"blip-image-captioning-base")


    train_dataloader = create_data_loader(os.path.join(opt.data, "train"), processor)
    val_dataloader = create_data_loader(os.path.join(opt.data, "val"), processor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    # 保存最佳模型
    best_val_loss = float("inf")
    for epoch in range(50):

        for step, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)

            loss = outputs.loss
            # 打印验证信息
            if step % 10 == 0:
                print(f"Train Epoch: {epoch}, Step: {step}, Loss: {loss.item():.5f}")
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            total_val_loss = 0
            num_batches = len(val_dataloader)
            for step, batch in enumerate(val_dataloader):
                # 将图像和文本编码
                input_ids = batch.pop("input_ids").to(device)
                pixel_values = batch.pop("pixel_values").to(device)

                outputs = model(input_ids=input_ids,
                                pixel_values=pixel_values,
                                labels=input_ids)

                val_loss = outputs.loss.item()
                total_val_loss += val_loss

            avg_val_loss = total_val_loss / num_batches
            print(f"Val Epoch: {epoch},   Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"------------   Saving best model  ----------- loss: {best_val_loss:.4f}")
                model.save_pretrained(os.path.join(save_dir, "best"))
                # torch.save(model.state_dict(), "../runs_yolov5/blip_best_model.pt")
        if epoch % 3 == 0:
            # 保存模型
            model.save_pretrained(os.path.join(save_dir, "epoch"))
            # torch.save(model.state_dict(), "../runs_yolov5/blip_model.pt")


def create_data_loader(data_dir, processor):
    print(f"Loading dataset {data_dir} ...")
    dataset = load_dataset("imagefolder", data_dir=data_dir, split="train")
    train_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    return train_dataloader


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
