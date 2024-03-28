import argparse
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # EfficientTeacher root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from transformers import AutoProcessor, BlipForConditionalGeneration
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


def main(opt):
    from datasets import load_dataset
    processor = AutoProcessor.from_pretrained(r"blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        r"blip-image-captioning-base")

    dataset = load_dataset("imagefolder", data_dir=os.path.join(opt.data, "train"), split="train")
    train_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    for epoch in range(50):
        loss_val = 0
        print("Epoch:", epoch, ",loss_val:", loss_val)
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)

            loss = outputs.loss
            loss_val = loss.item()
            print(loss_val)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
