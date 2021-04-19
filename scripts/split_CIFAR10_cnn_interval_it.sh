GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_it_test
REPEAT=1
mkdir -p $OUTDIR


python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
       --eps_val 1 --eps_epoch 20 --eps_max 1 \
       --kappa_epoch 10 --schedule 20 \
       | tee ${OUTDIR}/test1.log


# --eps_val 0.1 0.1 0.8 0.05 0.01 acc 70%
# --eps_val 0 acc 75%