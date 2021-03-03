GPUID=$1
OUTDIR=outputs/split_MNIST_interval_it
REPEAT=10
mkdir -p $OUTDIR


# Incremental Task
python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 0 \
       --first_split_size 2 --other_split_size 2 --schedule 20 --batch_size 128 --model_name MLP400 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet --lr 0.001 \
       --kappa_epoch 6 --eps_epoch 20 --eps_val 0.1 --eps_max 0.1 \
       | tee ${OUTDIR}/IN_Adam_01.log

python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 0 \
       --first_split_size 2 --other_split_size 2 --schedule 20 --batch_size 128 --model_name MLP400 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet --lr 0.001 \
       --kappa_epoch 6 --eps_epoch 20 --eps_val 0.1 --eps_max 0.1 --clipping \
       | tee ${OUTDIR}/IN_Adam_01_clipping.log

