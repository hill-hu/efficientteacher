import json
import os



from angel.utils import read_img_file, save_img_file
import numpy as np
from pathlib import Path


def stats_matrix(matrix, names, label_names, total):
    print("model names:", names)
    print("dataset names:", label_names)
    stats = []

    for i in range(len(names)):
        for j in range(len(names)):
            num = int(matrix[i][j])
            for k in range(num):
                stat = {"label": int(i), "predict": int(j)}
                stats.append(stat)

    # print(stats)
    stats_result(stats, names, label_names, total)


def stats_result(stats, names, label_names, total, save_dir=None):
    print("model names:", names)
    print("dataset names:", label_names)

    miss_count = 0
    for i in range(len(stats)):
        stat = stats[i]

        label_cls = str(label_names[stat['label']])
        if stat['predict'] < 0:
            miss_count += 1
            predict_cls = None
        else:
            predict_cls = str(names[stat['predict']])
        diff, precision = cal_diff(label_cls, predict_cls)

        stat.update({"diff": diff, "precision": precision,
                     "label_cls": label_cls, "predict_cls": predict_cls})

    print("miss count: ", miss_count, ",miss rate: ", miss_count / total)
    step, _ = cal_diff(str(label_names[1]), str(label_names[0]))
    print("step:", step)
    for i in range(0, 3):
        diff_stats = [stat for stat in stats if stat["diff"] <= i * step]
        diff_count = len(diff_stats)
        print(f"diff <={i}:", diff_count, f"/{total} ,rate:{diff_count / total}")

    if save_dir:
        diff_stats = [stat for stat in stats if abs(stat['predict'] - stat['label']) > 2]
        export_files(save_dir, diff_stats, "fail")
        diff_stats = [stat for stat in stats if abs(stat['predict'] - stat['label']) <= 2]
        export_files(save_dir, diff_stats, "ok")
    for p in [0.7, 0.8, 0.9, 1.0]:
        top_p = len([stat for stat in stats if p <= stat["precision"] <= 1])
        print(f"precision >={p}:", top_p, f"/{total} ,rate:{top_p / total}")

    return stats


def cal_diff(label, predict):
    # cal  abs diff and precision
    if predict is None:
        return 999, 0
    label = str(label).replace("mm", "").replace(">", "")
    predict = predict.replace("mm", "").replace(">", "")
    diff = int(label) - int(predict)
    return abs(diff), 1 - diff / int(label)


def read_dataset(dataset):
    #
    img_files = []
    label_names = dataset['names']
    val_path = dataset['val']
    with open(val_path, encoding='utf-8') as file:
        img_files += file.readlines()
    # load test data
    img_files = [line.replace('\n', "") for line in img_files]
    labels = []
    for line in img_files:
        label_file = line.replace("images", "labels").replace(".jpg", ".txt")
        with open(label_file, encoding='utf-8') as file:
            labels += file.readlines()
    labels = [int(label.split(" ")[0]) for label in labels]
    print(f"load {len(img_files)} data from {val_path},labels={len(labels)},names={label_names}")
    return label_names, labels, img_files


def stats_json(json_file, labels, img_files, label_names, model_names):
    # stats from  "predictions.json" file
    stats_cache = {}
    for i in range(len(labels)):
        img_id = os.path.split(img_files[i])[1].replace(".jpg", "")
        stat = {"label": labels[i], "predict": -1, "img_file": img_files[i], "image_id": img_id,
                "bbox": [100, 100, 100, 100]}
        stats_cache[img_id] = stat

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
        # {"image_id": "3056_1572", "category_id": 8, "bbox": [536.373, 34.919, 499.416, 665.0], "score": 1.0}
        for i in range(len(data)):
            image_id = str(data[i]["image_id"])
            stat = stats_cache.get(image_id, None)
            if not stat:
                print("image_id is missing:", image_id)
                continue

            if stat['predict'] >= 0:
                continue
            stat['predict'] = data[i]["category_id"]
            stat['bbox'] = data[i]["bbox"]
    stats = list(stats_cache.values())
    save_dir = os.path.split(json_file)[0]
    stats_result(stats, model_names, label_names, len(labels), save_dir)
    return stats


import cv2


def export_files(save_dir, top_diff, tag):
    output_folder = os.path.join(save_dir, "diff_" + tag)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for stat in top_diff:
        bbox = stat['bbox']
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])
        color = (255, 0, 0)
        bbox_img = read_img_file(stat["img_file"])
        bbox_img = cv2.rectangle(bbox_img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(bbox_img, "P:" + str(stat['predict_cls']), (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        cv2.putText(bbox_img, "G:" + str(stat['label_cls']), (xmin, ymax - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        save_img_file(os.path.join(output_folder, stat["image_id"] + ".jpg"), bbox_img)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """
    override mAp compute method
    @see utils.metrics.ap_per_class
    """
    from utils.metrics import compute_ap
    from utils.metrics import plot_pr_curve, plot_mc_curve
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        # union up cls and low cls
        if c > 0:
            i = i + (pred_cls == c - 1)
        if c < int(len(names) - 2):
            i = i + (pred_cls == c + 1)
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):

                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    # print('px:', px)
    # print('px shape:', px.shape)
    # print(i)
    # print(f1.shape)
    cls_thr = []
    for index in range(f1.shape[0]):
        # for f1_c in f1:
        f1_c = f1[index, :]
        # print('f1_c:', f1_c)
        c_i = f1_c.argmax()
        # print('c_i:', c_i)
        cls_thr.append(px[c_i])
    # print('cls_thr:', cls_thr)

    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32'), cls_thr


if __name__ == '__main__':
    data = r'E:\medical\depth\efficientteacher\configs\sup\custom\datasets\polyp_size_v2.yaml'
    save_dir = r'E:\medical\depth\efficientteacher\runs\val\exp60'
    label_names, labels, img_files = read_dataset(data)
    results = stats_json(os.path.join(save_dir, "best_predictions.json"),
                         labels, img_files, label_names, label_names)

