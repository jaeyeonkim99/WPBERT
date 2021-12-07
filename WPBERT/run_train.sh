OUTPUT_DIR=""
TRAIN_DATA_PATH=""
VALIDATION_DATA_PATH=""
PHONE_TYPE=""

python3 train_WPBERT.py --do_train --overwrite_output_dir --local_rank 0 &
python3 train_WPBERT.py --do_train --overwrite_output_dir --local_rank 1 &

