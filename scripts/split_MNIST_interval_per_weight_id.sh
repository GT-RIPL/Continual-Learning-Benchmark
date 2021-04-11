GPUID=$1
OUTDIR=outputs/split_MNIST_interval_id_per_weight
REPEAT=1
mkdir -p $OUTDIR

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 12 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --kappa_epoch 4 --eps_epoch 12 --eps_val 10 3.3 1.1 0.36 0.12 \
#       --eps_max 0 --clipping | tee ${OUTDIR}/in_pw_random.log

python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
       --eps_val 30 9.9 3.3 1.08 0.36 --eps_epoch 12 --eps_max 0 \
       --kappa_epoch 4 --schedule 12 \
       | tee ${OUTDIR}/in_pw_random1.log

