GPUID=$1
OUTDIR=outputs/split_MNIST_interval_id_per_weight
REPEAT=2
mkdir -p $OUTDIR

rm -rf tb_runs/*

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 12 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --kappa_epoch 4 --eps_epoch 12 --eps_val 10 3.3 1.1 0.36 0.12 \
#       --eps_max 0 --clipping | tee ${OUTDIR}/in_pw_random.log

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 10 --eps_epoch 8 --eps_max 0 \
#       --kappa_epoch 4 --schedule 8 \
#       | tee ${OUTDIR}/in_pw_random2.log

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 6 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 4 --schedule 12 \
#       | tee ${OUTDIR}/in_pw_random3.log

python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
       --eps_val 16 4 3 1 0.5 --eps_epoch 12 10 10 10 8 --eps_max 0 \
       --kappa_epoch 1 --schedule 12 10 10 10 8 \
       | tee ${OUTDIR}/in_pw_random4.log



# 16 4 3 1 0.5 - 80%
# 16 4 3 1 1
# --eps_val 15 4 3 1 1 81% - z exp.sum(dim=1)[:, None] BT i --eps_per_model
# --eps_val 100 10 1 0.1 0.01 acc 75% - z exp.sum(dim=1)[:, None] BT
# --eps_val 6000 2000 666 222 111

