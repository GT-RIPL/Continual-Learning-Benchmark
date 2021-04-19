GPUID=$1
OUTDIR=outputs/split_MNIST_interval_it_per_weight
REPEAT=3
mkdir -p $OUTDIR


#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 \
#       --first_split_size 2 --other_split_size 2 --schedule 12 --batch_size 100 --model_name MLP400 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet --lr 0.001 \
#       --kappa_epoch 4 --eps_epoch 12 --eps_val 5 1.8 0.5 0.2 0.1 --eps_max 10 --clipping \
#       | tee ${OUTDIR}/in_pw_zeros.log



python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
       --force_out_dim 0 --first_split_size 2 --other_split_size 2 --model_name MLP400 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 100 --lr 0.001 --clipping \
       --eps_val 3 --eps_epoch 8 --eps_max 0 \
       --kappa_epoch 1 --schedule 8 \
       | tee ${OUTDIR}/in_pw_zeros1.log


# --eps_val 10 3.3 1.1 0.36 0.12 acc 98%
# --eps_val 15 5.4 1.5 0.6 0.3  acc 97%
# --eps_val 4000 500 100 20 2  acc 97%
# --eps_val 10000 1000 300 80 10