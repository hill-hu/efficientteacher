import base64
import os
import random

__author__ = 'hill'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

import cv2
from datetime import datetime
from angel import logger
from angel.utils import save_img_file


class Plot:
    @staticmethod
    def transparent_overlay(src, overlay):

        h, w, _ = overlay.shape  # Size of foreground

        rows, cols, _ = src.shape  # Size of background Image
        pos = (random.randint(rows // 8, rows // 3), random.randint(cols // 8, cols // 3))
        y, x = pos[0], pos[1]  # Position of foreground/overlay image

        alpha_rate = random.randint(60, 100) * 0.01
        # loop over all pixels and apply the blending equation
        for i in range(h):
            for j in range(w):
                if x + i >= rows or y + j >= cols:
                    continue
                alpha = float(overlay[i][j][3] / 255.0) * alpha_rate  # read the alpha channel
                src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]

        return src

    @staticmethod
    def plot_cam(img, cam, out_put_dir, size=224, **img_param):
        cam = cv2.resize(cam, (size, size))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam <= 0.6)] = 0
        opacity = img_param.get("opacity", 0.4)
        weight_opacity = 0.2
        out = cv2.addWeighted(img, opacity, heatmap, weight_opacity, 0)

        # blank_img = np.zeros((224, 224, 4))
        heatmap_a = np.concatenate((heatmap, np.zeros((size, size, 1))), axis=2)
        bbox_out = heatmap_a.copy()
        bbox_out[np.where(cam > 0.6)] = 255

        bbox_out = cv2.addWeighted(bbox_out, 0.2, heatmap_a, 0.8, 0)
        retval, buffer = cv2.imencode('.png', bbox_out)
        cam_img = str(base64.b64encode(buffer)).replace("b'", 'data:image/png;base64,').replace("'", "")
        file_path = Plot.save_img(out, out_put_dir)
        return cam_img, file_path

    @staticmethod
    def convert_color(color_str):
        color_dict = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255)}
        return color_dict.get(color_str, (255, 0, 0))

    @staticmethod
    def plot_cam_bbox(src_img, rects, label, out_put_dir, color=(0, 255, 0), size=720, line_weight=2, color_str=None):
        """
        color 为bgr的模式

        从v5.0.6 开始注释掉置信度显示，若后续版本需要置信度，可以打开

        """

        if color_str is not None:
            color = Plot.convert_color(color_str)
        img = src_img.copy()
        bbox_img = cv2.resize(img, (size, size))
        scale = size / 224
        for rect in rects:
            x, y, x2, y2 = int(rect[0] * scale), int(rect[1] * scale), int(rect[2] * scale), int(rect[3] * scale)
            cv2.rectangle(bbox_img, (x, y), (x2, y2), color, line_weight)
            # 显示置信度，
            # if label:
            #     cv2.putText(bbox_img, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        cam_file_path = Plot.save_img(bbox_img, out_put_dir)

        return cam_file_path

    @staticmethod
    def plot_overlay(src_img, label, file_path, size=720, line_weight=2, show_confidence=False):
        if src_img is None:
            logger.warning("src_img is none")
            return
        color_str = label.overlay_color
        rects = label.overlay_box
        color = Plot.convert_color(color_str)
        img = src_img.copy()
        bbox_img = cv2.resize(img, (size, size))
        scale = size / 224
        for rect in rects:
            x, y, x2, y2 = int(rect[0] * scale), int(rect[1] * scale), int(rect[2] * scale), int(rect[3] * scale)
            cv2.rectangle(bbox_img, (x, y), (x2, y2), color, line_weight)
            if show_confidence:
                text = "0.%s" % int(label.confidence)
                cv2.putText(bbox_img, str(text), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        save_img_file(file_path, bbox_img)

    @staticmethod
    def cal_cam_bbox(cam):
        cam = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam <= 0.6)] = 0
        heatmap[np.where(heatmap > 0)] = 255
        gray_image = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        x, y, w, h = cv2.boundingRect(gray_image)
        max_width = int(224 * 0.5)
        if w > max_width:
            x += (w - max_width) // 2
            w = max_width

        if h > max_width:
            y += (h - max_width) // 2
            h = max_width
        if y < 4:
            y = 4
        out_x, out_y, out_x1, out_y1 = Plot.filter_cam_bbox(cam, [x, y, w, h])
        return [out_x, out_y, out_x1, out_y1]

    @staticmethod
    def filter_cam_bbox(cam, source_box):
        x, y, w, h = source_box
        limit_x = 30
        limit_y = 10
        ix = limit_x
        iy = limit_y
        ix1 = cam.shape[0] - limit_x
        iy1 = cam.shape[1] - limit_y

        if x < ix or y < iy or x + w > ix1 or y + h > iy1:
            w = int(cam.shape[0] / 2)
            h = int(cam.shape[1] / 2)
            out_x = int((cam.shape[0] - w) / 2)
            out_x1 = out_x + w
            out_y = int((cam.shape[1] - h) / 2)
            out_y1 = out_y + h
        else:
            out_x = x
            out_x1 = x + w
            out_y = y
            out_y1 = y + h
        return [out_x, out_y, out_x1, out_y1]

    @staticmethod
    def save_img(bbox_img, out_put_dir):
        rand = random.randint(1, 1000)
        file_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3] + "_" + str(rand) + ".jpg"
        file_path = os.path.abspath(out_put_dir + '/' + file_name)
        save_img_file(file_path, bbox_img)
        logger.info("export img %s " % file_path)
        return file_path

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.round(cm, 2)
            print("Normalized confusion matrix")

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Predict')
        plt.xlabel('True')

    @staticmethod
    def show_matrix(y_pred, y_true, classes_count, out_put_dir):
        classes = [str(x) for x in range(classes_count)]
        cnf_matrix = confusion_matrix(y_true, y_pred, classes)
        # Plot non-normalized confusion matrix
        plt.figure()
        plt.figure(figsize=(4, 4))

        Plot.plot_confusion_matrix(cnf_matrix, classes=classes,
                                   title='Confusion matrix')

        rand = random.randint(1, 1000)
        file_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3] + "_" + str(rand) + ".png"
        file_path = os.path.abspath(out_put_dir + '/' + file_name)
        plt.savefig(file_path)
        logger.info("save plot image to %s " % file_path)

        err = 0
        for i in range(0, len(y_pred)):
            if y_pred[i] != y_true[i]:
                err += 1

        logger.info("accurracy:{}".format(1 - err * 1.0 / len(y_pred)))
        return file_path

    @staticmethod
    def get_img_margin(img_size, crop_size):
        src_w = img_size[0]
        src_h = img_size[1]

        if not crop_size:
            return 1, src_w - 1, 1, src_h - 1

        w = crop_size["width"]
        h = crop_size["height"]
        if w == 0 or h == 0:
            return 1, src_w - 1, 1, src_h - 1

        if w > h:
            new_h = int(src_w * h / w)
            y1_del = int((src_h - new_h) / 2)
            y2_del = y1_del + new_h
            x_min = 1
            x_max = src_w - 1
            y_min = y1_del
            y_max = y2_del
        else:
            new_w = int(src_h * w / h)
            x1_del = int((src_w - new_w) / 2)
            x2_del = x1_del + new_w
            y_min = 1
            y_max = src_h - 1
            x_min = x1_del
            x_max = x2_del

        return x_min, x_max, y_min, y_max

    @staticmethod
    def expand_box(bbox, ratio=1.2, image_size=(224, 224), img_params=None):

        crop_size = img_params.get("detect_size", None)
        x_min, x_max, y_min, y_max = Plot.get_img_margin(image_size, crop_size)

        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        w_box = x2 - x1
        h_box = y2 - y1
        x0 = x1 + (x2 - x1) / 2
        y0 = y1 + (y2 - y1) / 2

        # 重新计算box坐标
        delta_w = max(min(w_box * ratio / 2, 50), 10)
        delta_h = max(min(h_box * ratio / 2, 50), 10)

        new_x1 = int(x0 - delta_w)
        new_x1 = max(new_x1, x_min)

        new_x2 = int(x0 + delta_w)
        new_x2 = min(new_x2, x_max)

        new_y1 = int(y0 - delta_h)
        new_y1 = max(new_y1, y_min)

        new_y2 = int(y0 + delta_h)
        new_y2 = min(new_y2, y_max)

        return [new_x1, new_y1, new_x2, new_y2]

    @staticmethod
    def expand_overlay_box(overlay_box, ratio=1.2, image_size=(224, 224), img_params=None):
        out_overlay_box = []
        if not overlay_box:
            return out_overlay_box
        for box in overlay_box:
            # 目标较小时，拓展外框，使更容易分辨
            if box[2] - box[0] < 224 * 0.4 and box[3] - box[1] < 224 * 0.4:
                out_overlay_box.append(Plot.expand_box(box, ratio, image_size, img_params=img_params))
            else:
                out_overlay_box.append(box)
        return out_overlay_box
