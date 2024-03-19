import torch
from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import torch.nn as nn
from trainer.trainer import Trainer
from trainer.ssod_trainer import SSODTrainer
from configs.defaults import get_cfg
from utils.general import increment_path, check_git_status, check_requirements, \
    print_args, set_logging
from pathlib import Path
import logging
import os
import argparse
from utils.callbacks import Callbacks
from utils.torch_utils import select_device
import sys
from datetime import timedelta
import val
# 定义目标检测模型
from models.detector.yolo import Model

device = torch.device('cuda:0')


def test():
    import torch

    # 定义一个包含两个卷积层和一个全连接层的模型
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=32 * 3 * 3, out_features=10)
    )

    # 查看模型的输入输出形状
    input_shape = torch.randn((16, 6, 3, 3))
    output_shape = model(input_shape).shape

    print(f"输入形状：{input_shape.shape}")
    print(f"输出形状：{output_shape}")


class FCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 定义全连接层
        self.fc1 = torch.nn.Linear(in_features=25200 * 6, out_features=1024, dtype=torch.half)

        self.fc2 = torch.nn.Linear(in_features=1024, out_features=1)

    def forward(self, x):
        # 将图像展平
        x = x.view(-1, 25200 * 6)

        # 全连接层
        x = self.fc1(x)
        x = torch.nn.ReLU()(x)
        x = self.fc2(x)

        return x


def load_model():
    ckpt = torch.load(r'E:\medical\depth\runs\exp15_polyp_yolov5\best.pt', map_location=device)  # load checkpoint
    model = ckpt['model']
    return model


def train(model):
    # 定义分类模型
    classifier = FCNN().to(device)
    classifier.cuda().half()

    # 微调分类模型
    batch_size = 32
    train_dataset = datasets.ImageFolder(root=r"E:\medical\depth\dataset\classify", transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((640, 640)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    # 定义损失函数
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    input_shape = None
    for epoch in range(10):
        for images, labels in train_loader:
            images = images.cuda().half()
            # 训练模型

            # 前向传播
            outputs = model(images)
            inputs = outputs[0]
            classify_input = inputs.view(batch_size, -1)
            if input_shape is None:
                input_shape = inputs.shape
                print("yolo output shape:", input_shape, ",classify_input:", classify_input.shape)
            # 梯度清零
            optimizer.zero_grad()
            logits = classifier(classify_input)
            # 计算损失
            _labels = labels.view(-1, 1).to(device).cuda().half()
            print("classify output shape:", logits.shape, ", values:", logits)
            print("labels shape:", _labels.shape, ", values:", _labels)

            loss = criterion(logits, _labels)
            print("loss:", loss.item())

            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()


if __name__ == "__main__":
    loss = nn.BCEWithLogitsLoss()
    input = torch.Tensor([-0.3418, 0, 0.3376])
    target = torch.Tensor([0., 1., 0.])
    print(input)
    print(target)
    input_cls = torch.argmax(input)
    target_cls = torch.argmax(target)
    output = loss(input, target)
    print("input cls:", input_cls, "target cls:", target_cls)
    if input_cls != target_cls:
        if abs(input_cls - target_cls) == 1:
            output *= 0.9

    print("loss:", output.item())
