python src/medclip/run_medclip.py \
    --output_dir ./snapshots/vision_augmented_biobert \
    --text_model_name_or_path="allenai/scibert_scivocab_uncased" \
    --vision_model_name_or_path="openai/clip-vit-base-patch32" \
    --tokenizer_name="allenai/scibert_scivocab_uncased" \
    --train_file="data/train_dataset_new.json" \
    --validation_file="data/valid_dataset_new.json" \
    --do_train --do_eval \
    --num_train_epochs="40" --max_seq_length 128 \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --preprocessing_num_workers 4
#    --push_to_hub
