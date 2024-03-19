import cv2
import os
import traceback

import numpy as np


def read_img_file(path):
    try:
        with open(path, 'rb') as img_file:
            bytes = img_file.read()
            nparr = np.frombuffer(bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
    except:
        traceback.print_exc()
    return None


def save_img_file(path, img):
    try:
        with open(path, 'wb') as img_file:
            data = cv2.imencode('.jpg', img)[1]
            img_file.write(data)
    except:
        traceback.print_exc()
    return None


def fetch_all_files(from_dir, followlinks=True, file_exts=None, exclude_file_exts=None):
    """
    获取目录下所有文件
    """
    all_files = []
    for root, dirs, files in os.walk(from_dir, followlinks=followlinks):
        for name in files:
            if file_exts:
                _, ext = os.path.splitext(name)
                if ext not in file_exts:
                    print("exclude file %s,%s" % (name, ext))
                    continue

            if exclude_file_exts:
                _, ext = os.path.splitext(name)
                if ext in exclude_file_exts:
                    print("exclude file %s,%s" % (name, ext))
                    continue

            path_join = os.path.join(root, name)
            all_files.append(path_join)

    print("fetch_all_files count=%s" % len(all_files))
    return all_files
