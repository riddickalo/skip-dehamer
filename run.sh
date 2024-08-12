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

# test.sh example
# pipenv run python test.py --gpu_device 0 --dataset 'Sate1K_Thick' --model_size SkipDehamer \
#     --model_dir '2024-06-16_16' --model_name 'best_psnr.pth'