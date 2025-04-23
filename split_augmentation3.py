# yolov9t_augmentation.py

import os
import random
import yaml
import cv2
import numpy as np

def horizontal_flip_image(img):
    return cv2.flip(img, 1)

def horizontal_flip_labels(label_lines):
    flipped_labels = []
    for line in label_lines:
        cls, x_center, y_center, width, height = map(float, line.strip().split())
        x_center_flipped = 1.0 - x_center
        flipped_line = f"{int(cls)} {x_center_flipped:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        flipped_labels.append(flipped_line)
    return flipped_labels

def color_jitter_image(img):
    """밝기(brightness), 대비(contrast) 랜덤 조정"""
    img = img.astype(np.float32) / 255.0

    brightness = 0.8 + random.random() * 0.4  # 0.8 ~ 1.2
    contrast = 0.8 + random.random() * 0.4    # 0.8 ~ 1.2

    img_mean = np.mean(img, axis=(0, 1), keepdims=True)
    img = (img - img_mean) * contrast + img_mean
    img = img * brightness
    img = np.clip(img, 0, 1)

    img = (img * 255).astype(np.uint8)
    return img

def rename_images_and_labels(images_dir, labels_dir):
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    image_basenames = {os.path.splitext(f)[0]: f for f in image_files}
    label_basenames = {os.path.splitext(f)[0]: f for f in label_files}

    common_keys = sorted(list(set(image_basenames.keys()) & set(label_basenames.keys())))

    for idx, key in enumerate(common_keys, 1):
        img_file = image_basenames[key]
        lbl_file = label_basenames[key]

        img_ext = os.path.splitext(img_file)[1]
        new_img_name = f"{idx:03d}{img_ext}"
        new_lbl_name = f"{idx:03d}.txt"

        os.rename(os.path.join(images_dir, img_file), os.path.join(images_dir, new_img_name))
        os.rename(os.path.join(labels_dir, lbl_file), os.path.join(labels_dir, new_lbl_name))

def offline_augmentation(images_dir, labels_dir, color_aug_prob=0.3): # 확률 적용
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    image_files = sorted([f for f in image_files if 'augmented' not in f])

    total_augmented = 0
    max_augmented_images = len(image_files)  # 예를 들어 480개

    for img_file in image_files:
        if total_augmented >= max_augmented_images:
            break

        img_idx = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        lbl_path = os.path.join(labels_dir, f"{img_idx}.txt")

        aug_img_path = os.path.join(images_dir, f"{img_idx}_augmented.jpg")
        aug_lbl_path = os.path.join(labels_dir, f"{img_idx}_augmented.txt")

        if os.path.exists(aug_img_path) and os.path.exists(aug_lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # 1. 기본적으로 좌우반전 적용
        aug_img = horizontal_flip_image(img)

        # 2. color jitter를 color_aug_prob 확률로 추가 적용
        if random.random() < color_aug_prob:
            aug_img = color_jitter_image(aug_img)

        cv2.imwrite(aug_img_path, aug_img)

        with open(lbl_path, 'r') as f:
            label_lines = f.readlines()

        flipped_labels = horizontal_flip_labels(label_lines)

        with open(aug_lbl_path, 'w') as f:
            for line in flipped_labels:
                f.write(line + '\n')

        total_augmented += 1

def split_dataset(base_dir, dataset_name, iter_num):
    images_dir = os.path.join(base_dir, 'images')
    output_dir = base_dir

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    image_files.sort()

    random.seed(iter_num)
    files = image_files.copy()
    random.shuffle(files)

    normal_files = [f for f in files if 'augmented' not in f]
    augmented_files = [f for f in files if 'augmented' in f]

    n_total = len(normal_files)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    n_test = n_total - n_train - n_val

    train_files = normal_files[:n_train]
    val_files = normal_files[n_train:n_train + n_val]
    test_files = normal_files[n_train + n_val:]

    full_train_files = train_files.copy()
    for f in train_files:
        idx = os.path.splitext(f)[0]
        aug_f = f"{idx}_augmented.jpg"
        if aug_f in augmented_files:
            full_train_files.append(aug_f)

    splits = {
        'train': full_train_files,
        'val': val_files,
        'test': test_files
    }

    for split, split_files in splits.items():
        txt_filename = f'{split}_iter_{iter_num:02d}.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, 'w') as f:
            for file in split_files:
                f.write(f'Datasets/{dataset_name}/images/{file}\n')

    yaml_data = {
        'names': ['airplane'],
        'nc': 1,
        'path': f'Datasets/{dataset_name}',
        'train': f'train_iter_{iter_num:02d}.txt',
        'val': f'val_iter_{iter_num:02d}.txt',
        'test': f'test_iter_{iter_num:02d}.txt'
    }
    yaml_filename = f'data_iter_{iter_num:02d}.yaml'
    yaml_path = os.path.join(output_dir, yaml_filename)
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

def augment_train_images(dataset_root, iter_num):
    """
    메인 함수: dataset_root ('Datasets/Airplane'), iter_num (1~10) 받아서 실행
    """
    images_dir = os.path.join(dataset_root, 'images')
    labels_dir = os.path.join(dataset_root, 'labels')

    if iter_num == 1:
        rename_images_and_labels(images_dir, labels_dir)
        offline_augmentation(images_dir, labels_dir, color_aug_prob=0.5)

    split_dataset(dataset_root, 'Airplane', iter_num)
