#!/bin/bash

#export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
#export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
#export TRAIN_DIR="../example_data/images"
#export OUTPUT_DIR="../sdxl-sm-test"

# Clone the repo:

git clone https://github.com/TencentARC/BrushNet.git

# We recommend you first use conda to create virtual environment, and install pytorch following official instructions. For example:

# conda create -n diffusers python=3.9 -y
# conda activate diffusers
# python -m pip install --upgrade pip
# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Then, you can install diffusers (implemented in this repo) with:

# cd BrushNet

pip install -e BrushNet

# After that, you can install required packages thourgh:

pip install -r requirements_brushnet.txt

chmod +x ./s5cmd


# TRAIN_DIR=$TRAIN_DIR/pokeman_images

accelerate launch train_brushnet.py \
--pretrained_model_name_or_path models/realisticVisionV60B1_v51VAE \
--brushnet_model_name_or_path models/random_mask_brushnet_ckpt \
--output_dir $OUTPUT_DIR \
--train_data_dir data/heguan_reformed \
--validation_image data/heguan_reformed/test_image.jpg \
--validation_mask data/heguan_reformed/test_mask.jpg \
--num_train_epochs 2 \
--resolution 256 \
--learning_rate 1e-5 \
--train_batch_size 2 \
--num_per_tar_file 500 \
--tracker_project_name brushnet \
--report_to tensorboard \
--resume_from_checkpoint latest \
--validation_steps 500 \
--checkpoints_total_limit 5 \
--checkpointing_steps 1000