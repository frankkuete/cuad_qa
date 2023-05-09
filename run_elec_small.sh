python run_qa_electra.py \
  --model_name_or_path google/electra-small-discriminator \
  --dataset_name cuad \
  --do_train \
  --do_predict \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 4 \
  --max_seq_length 512 \
  --max_answer_length 512 \
  --doc_stride 256 \
  --output_dir ../train_models/electra-small \
  --overwrite_output_dir \
  --save_steps 10000 \
  --version_2_with_negative