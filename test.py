from multiprocessing import freeze_support

import logging
import os
import argparse

import sys
from datetime import timedelta




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='model weights path')
    parser.add_argument("--data", type=str, help="data.yaml path ")

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def cal_diff(label, predict):
    # cal  abs diff and precision
    if predict is None:
        return 999, 0
    label = label.replace("mm", "").replace(">", "")
    predict = predict.replace("mm", "").replace(">", "")
    diff = int(label) - int(predict)
    return abs(diff), 1 - diff / int(label)


if __name__ == '__main__':
    freeze_support()
    # Load a model
    opts = parse_opt()
    print(opts)

    lines = []
    with open(opts.data) as file:
        lines += file.readlines()
    # load test data
    lines = [line.replace('\n', "") for line in lines]
    labels = []
    for line in lines:
        label_file = line.replace("images", "labels").replace(".jpg", ".txt")
        with open(label_file) as file:
            labels += file.readlines()
    labels = [int(label.split(" ")[0]) for label in labels]
    print(f"load {len(lines)} data from {opts.data},labels={len(labels)}")

    model = YOLO(opts.weights)  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    stats = []
    names = model.names
    print("class names :", names)
    batch_n = 0
    batch_size = 64
    while batch_n < len(labels) / batch_size:
        results = model(lines[batch_n * batch_size:(batch_n + 1) * batch_size])
        batch_n += 1
        for i, result in enumerate(results):
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            probs = result.probs  # Probs object for classification outputs
            # result.show()  # display to screen
            # result.save(filename='result.jpg')  # save to disk
            size = 0
            predict = None
            names = result.names

            if len(boxes) > 0:
                predict = names[int(boxes[0].cls)]
                # print("class_name:", predict)
            else:
                print("predict is miss:", result.path)

            label_cls = names[labels[i]]
            diff, precision = cal_diff(label_cls, predict)
            stat = {"path": result.path, "diff": diff, "precision": precision,
                    "label": label_cls, "predict": predict}
            stats.append(stat)

    total = len(lines)

    for i in range(0, 3):
        top_diff = len([stat for stat in stats if stat["diff"] <= i])
        print(f"diff <={i}:", top_diff, f"/{total} ,rate:{top_diff / total}")

    for p in [0.7, 0.8, 0.9, 1.0]:
        top_p = len([stat for stat in stats if stat["precision"] >= p])
        print(f"precision >={p}:", top_p, f"/{total} ,rate:{top_p / total}")
