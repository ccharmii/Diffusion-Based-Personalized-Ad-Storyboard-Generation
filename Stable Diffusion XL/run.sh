accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --train_data_dir="./data/train/dataset3" \
  --caption_column="caption" \
  --image_column="image" \
  --resolution=1024 \
  --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 \
  --checkpointing_steps=200 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="sdxl-lora-2" \
  --validation_prompt="A grayscale sketch of the morning sunlight enters from the top left through the window, and a person is sitting at a front table, drinking coffee, and working on an iPad" \
  --report_to="wandb"
