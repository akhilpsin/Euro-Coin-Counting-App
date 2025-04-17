import os
import random
import shutil

# Paths
base_dir = "euro_coins_dataset"
all_images_dir = os.path.join(base_dir, "all/images_all")  # where all images are
all_labels_dir = os.path.join(base_dir, "all/labels_all")  # where all labels are

output_img_dir = os.path.join(base_dir, "images")
output_lbl_dir = os.path.join(base_dir, "labels")

# Create folders
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_img_dir, split), exist_ok=True)
    os.makedirs(os.path.join(output_lbl_dir, split), exist_ok=True)

# Split ratio
split_ratio = 0.8

images = [f for f in os.listdir(all_images_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(images)

split_idx = int(len(images) * split_ratio)
train_images = images[:split_idx]
val_images = images[split_idx:]

# Function to move image + label
def move_data(img_list, split):
    for img_name in img_list:
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        src_img = os.path.join(all_images_dir, img_name)
        src_lbl = os.path.join(all_labels_dir, label_name)

        dst_img = os.path.join(output_img_dir, split, img_name)
        dst_lbl = os.path.join(output_lbl_dir, split, label_name)

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

move_data(train_images, 'train')
move_data(val_images, 'val')

print("✅ Dataset split and rearranged!")
