import os
from PIL import Image
import json
import random
import cv2
# import pickle
import webdataset as wds
import numpy as np

image_folder = "data/heguan-clothing-styles/images"
image_files = os.listdir(os.path.join(image_folder))
print(len(image_files), image_files[:5])

arms_mask_folder = "data/heguan-clothing-styles/inpainting_maskv3/arms_agnostic"
arms_mask_files = os.listdir(os.path.join(arms_mask_folder))
print(len(arms_mask_files), arms_mask_files[:5])

neck_mask_folder = "data/heguan-clothing-styles/inpainting_maskv3/neck_agnostic"
neck_mask_files = os.listdir(os.path.join(neck_mask_folder))
print(len(neck_mask_files), neck_mask_files[:5])

length_mask_folder = "data/heguan-clothing-styles/inpainting_maskv3/length_agnostic"
length_mask_files = os.listdir(os.path.join(length_mask_folder))
print(len(length_mask_files), length_mask_files[:5])

# caption_path = "data/heguan-clothing-styles/metadata_with_mask.v3.jsonl"
caption_path = "data/heguan-clothing-styles/metadata_with_mask.jsonl"

captions = []

with open(caption_path, 'r') as fin:
    for line in fin:
        captions.append(json.loads(line))

import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

df_captions = pd.DataFrame(captions)
df_captions.head()

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    # return ' '.join(str(x) for x in runs)
    return runs.tolist()


def rle2mask(mask_rle: str, label=1, shape=(512, 512)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction


def binarize_mask(mask):
    """
    Convert a NumPy array mask into a binary mask with values 0 and 1.
    1 - background, 0 - mask
    """
    binary_mask = 1.*(mask.sum(-1)>255)[:,:,np.newaxis]
    # print(binary_mask)
    # binary_mask = np.zeros_like(mask, dtype=np.uint8)
    # binary_mask[mask > 200] = 1
    return 1-binary_mask

def handle_caption(text):
    parts = text.split(',')
    
    styles = []
    for part in parts:
        part = part.strip()
        if '(' in part:
            styles.append(part.strip('(').strip(')').strip())
        elif ':' in part:
            kv = part.split(':')
            if len(kv) > 1:
                if kv[1].strip():
                    styles.append(part)
        else:
            styles.append(part)
    
    # random.shuffle(styles)

    return ', '.join(styles)
    


num_per_tar = 10
num_images = 50 # len(image_files)

# Create a WebDataset writer
for tar_id in range(num_images//num_per_tar+1):
    # Define the output dataset directory
    output_dir = f"data/heguan-reformed/{tar_id:03d}.tar"
    writer = wds.TarWriter(output_dir)

    for i, image_file in enumerate(image_files[tar_id*num_per_tar:(tar_id+1)*num_per_tar]):
        image_array = cv2.imread(os.path.join(image_folder, image_file))

        _, encoded_image = cv2.imencode('.jpg', image_array)
        byte_array = encoded_image.tobytes()
        file_path = f"data/heguan-reformed/{image_file.split('.')[0]}"
        # with open(file_path + ".image", 'wb') as fout:
        #     fout.write(byte_array)

        height, width = image_array.shape[0], image_array.shape[1]
        # with open(file_path + ".height", 'wb') as fout:
        #     # pickle.dump(height, fout)
        #     fout.write(str(height).encode())

        # with open(file_path + ".width", 'wb') as fout:
        #     # pickle.dump(str(width), fout)
        #     fout.write(str(width).encode())

        # print(type(image_array), image_array.shape)
        df_part = df_captions[df_captions['file_name'].str.endswith(image_file)].reset_index(drop=True)
        if df_part.shape[0] == 0:
            print(df_part)
            continue
            
        caption = handle_caption(df_part.loc[0, 'text'])
        # with open(file_path + ".caption", 'wb') as fout:
        #     # pickle.dump(df_part.loc[0, 'prompt'], fout)
        #     fout.write(str(df_part.loc[0, 'prompt']).encode())

        # mask
        arms_mask_path = os.path.join(arms_mask_folder, image_file)
        if os.path.exists(arms_mask_path):
            arms_mask_array = cv2.imread(arms_mask_path)
            # mask_array = np.array(mask)
            arms_mask_array_binary = binarize_mask(arms_mask_array)
            arms_mask_rle = mask2rle(arms_mask_array_binary)

        neck_mask_path = os.path.join(neck_mask_folder, image_file)
        if os.path.exists(neck_mask_path):
            neck_mask_array = cv2.imread(neck_mask_path)
            # mask_array = np.array(mask)
            neck_mask_array_binary = binarize_mask(neck_mask_array)
            neck_mask_rle = mask2rle(neck_mask_array_binary)

        length_mask_path = os.path.join(length_mask_folder, image_file)
        if os.path.exists(length_mask_path):
            length_mask_array = cv2.imread(length_mask_path)
            # mask_array = np.array(mask)
            length_mask_array_binary = binarize_mask(length_mask_array)
            length_mask_rle = mask2rle(length_mask_array_binary)

        segmentation = {'mask': [arms_mask_rle, neck_mask_rle, length_mask_rle]}

        sample = {
            "__key__": image_file.split('.')[0],
            "width": str(width).encode(),
            "height": str(height).encode(),
            "caption": caption.encode(),
            "image": byte_array,
            "segmentation": json.dumps(segmentation).encode(),
        }

        # print(sample)
        # Write the sample to the dataset
        writer.write(sample)

    # Close the writer
    writer.close()