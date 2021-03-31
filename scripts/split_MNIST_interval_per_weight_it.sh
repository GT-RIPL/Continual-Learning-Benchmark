GPUID=$1
OUTDIR=outputs/split_MNIST_interval_it_per_weight
REPEAT=10
mkdir -p $OUTDIR



python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 0 \
       --first_split_size 2 --other_split_size 2 --schedule 12 --batch_size 100 --model_name MLP400 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet --lr 0.001 \
       --kappa_epoch 4 --eps_epoch 12 --eps_val 5 1.8 0.5 0.2 0.1 --eps_max 10 --clipping \
       | tee ${OUTDIR}/in_pw_zeros.log


