GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_id_test
REPEAT=1
mkdir -p $OUTDIR


python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
       --eps_val 10 3.3 1.1 0.36 0.12 --eps_epoch 1 --eps_max 0 \
       --kappa_epoch 1 --schedule 1 \
       | tee ${OUTDIR}/test.log