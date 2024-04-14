#export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
#export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
#export TRAIN_DIR="../example_data/images"
#export OUTPUT_DIR="../sdxl-sm-test"

TRAIN_DIR=$TRAIN_DIR/pokeman_images

accelerate launch --config_file as_local_config.yaml train_text_to_image_sdxl_v2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$TRAIN_DIR --dataloader_num_workers=4 \
  --enable_xformers_memory_efficient_attention \
  --resolution=1024 --random_flip \
  --proportion_empty_prompts=0 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=100 \
  --use_8bit_adam \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="tensorboard" \
  --validation_prompt=" " \
  --validation_epochs 1000000 \
  --checkpointing_steps=50 \
  --output_dir=$OUTPUT_DIR 