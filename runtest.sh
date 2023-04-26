python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name cuad \
  --do_predict \
  --per_device_train_batch_size 20 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 512 \
  --max_answer_length 512 \
  --doc_stride 256 \
  --output_dir ./train_models/bert-base \
  --overwrite_output_dir \
  --save_steps 10000 \
  --version_2_with_negative \
  --max_train_samples 100 \
  --max_predict_samples 10 \


# for help type python run_qa.py --help 
#--dataset_name cuad \
#--train_file ./data/train_separate_questions.json \
#--test_file ./data/test.json \
# model_name_or_path --> Path to pretrained model or model identifier from huggingface.co/models
# dataset_name --> The name of the dataset to use (via the datasets library).
# do_train --> Whether to run training. (default: False)
# do_eval --> Whether to run eval on the dev set. (default: False)
# do_predict --> Whether to run predictions on the test set. (default: False)
# train_file --> The input training data file (a text file).
# validation_file -->An optional input evaluation data file to evaluate the perplexity on (a text file).
# test_file --> An optional input test data file to evaluate the perplexity on (a text file)
# max_seq_length --> The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
# max_train_samples --> For debugging purposes or quicker training, truncate the number of training examples to this 
# max_eval_samples --> For debugging purposes or quicker training, truncate the number of evaluation examples to this 
# max_predict_samples --> For debugging purposes or quicker training, truncate the number of prediction examples to this
# output_dir --> output directory
# overwrite_output_dir --> if set we will overwrite the output directory already created