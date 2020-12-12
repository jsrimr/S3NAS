# Directory settings
export STORAGE_BUCKET=gs://caplab_us_central1
export DATA_DIR=${STORAGE_BUCKET}/imagenet

export MODEL_NAME=mbconv_super_4_4_4_4_4_search_parsed_model
export MODEL_JSON=models/${MODEL_NAME}.json

export TEST_NAME=${MODEL_NAME}
export TEST_SAVE_FOLDER=${STORAGE_BUCKET}
export TRAIN_DIR=${TEST_SAVE_FOLDER}/${TEST_NAME}_train

# tpu
export TPU_NAME=train-parsed-model
export TPU_SETTINGS=--tpu=${TPU_NAME}
ctpu up --zone=us-central1-f --tf-version=1.14 --name=${TPU_NAME} -tpu-only

# Common settings
export TRAIN_BATCH_SIZE=1024
export EVAL_BATCH_SIZE=1000
export DROPOUT=0.2

export INPUT_IMAGE_SIZE=224

export PYTHONPATH=$PYTHONPATH:$(pwd)

# train result
export TRAIN_SETTINGS="--model_json_path=${MODEL_JSON} --input_image_size=$INPUT_IMAGE_SIZE --model_dir=$TRAIN_DIR --train_batch_size=$TRAIN_BATCH_SIZE --eval_batch_size=$EVAL_BATCH_SIZE --dropout_rate=$DROPOUT --data_dir=$DATA_DIR"
export TRAIN_LOG_FILE=${TEST_NAME}_train.log
python3 run/main.py --train_epochs=120 --epochs_per_eval=40 $TPU_SETTINGS $TRAIN_SETTINGS 2>>$TRAIN_LOG_FILE
python3 run/main.py --train_epochs=350 --epochs_per_eval=5 $TPU_SETTINGS $TRAIN_SETTINGS 2>>$TRAIN_LOG_FILE
gsutil cp ${TRAIN_LOG_FILE} ${TRAIN_DIR}/${TRAIN_LOG_FILE}
