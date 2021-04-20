GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_ic
REPEAT=1
mkdir -p $OUTDIR
python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 128 \
       --eps_val 1 0.5 0.5 0.5 0.5 --eps_epoch 20 --eps_max 0 \
       --kappa_epoch 10 --schedule 20 \
        | tee ${OUTDIR}/Adam.log

