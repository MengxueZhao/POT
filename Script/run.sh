MODEL=$1
GPU=$2
Train=$3
SAVE_PATH=$(pwd)/Output/${MODEL}

CUDA_VISIBLE_DEVICES=${GPU} nohup python -u main.py \
   --model_name ${MODEL} \
   --gpu_id ${GPU} \
   --do_train ${Train} \
   --mini_data \
   --mini_data_range "12,7,64,10,64,512,9" \
   --pos_neg_sample 20 \
   --pkl_dir $SAVE_PATH \
   --similar_neighbor 15 \
   --batch 4 \
   --print_seg 960 \
   --lr 0.00005 \
   --top_number 6 \
   > ./Script/log_${MODEL}_${Train}.file 2>&1 &



























