#! /bin/bash
if [ $1 == "--train" ]; then
    export LOG_DIR=$(date +%F_%H)

    if [ $2 == "--bg" ]; then
        echo "training will take place in background"
        if [ $3 == "-q"]; then
            echo "and quietly~"
            nohup ./train.sh > /dev/null 2>&1 &
        else
            echo "err_log ON"
            nohup ./train.sh > /dev/null 2>err_log.out &
        fi
    else
        echo "all logs will show on tty"
        ./train.sh
    fi
    exit 0
elif [ $1 == "--test" ]; then
    echo "Begin testing"
    ./test.sh
    exit 0
else
    echo "ERROR: Please specify run mode! ("--train" or "--test")"
    exit 2
fi


# train.sh example
# pipenv run python train.py --gpu_device 0 --dataset Sate1K_Thick --epochs 500  --notify_period 100 \
# 				--model_size SkipDehamer --batch_size 48 --accum_grad_step 1 --lr 0.0005 --num_workers 8 \
# 				--scheduler OneCycleLR --relu_type Leaky_ReLU --optimizer AdamW --bn_type GroupNorm --num_groups 3 \
# 				--grad_clip_type clip_grad_norm --clip_grad_max_norm 1.0 --use_crop --use_bn --use_grad_clip \
# 				--lambda_perceptual_loss 1.0 --lambda_l1_loss 10.0 --lambda_edge_loss 1.0 --lambda_ssim_loss 1.0 \
# 				--use_l1_loss --use_edge_loss

# test.sh example
# pipenv run python test.py --gpu_device 0 --dataset 'Sate1K_Thick' --model_size SkipDehamer \
#     --model_dir '2024-06-16_16' --model_name 'best_psnr.pth'