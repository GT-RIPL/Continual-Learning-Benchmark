GPUID=$1
OUTDIR=outputs/split_MNIST_interval_ic
REPEAT=10
mkdir -p $OUTDIR

# Incremental Task
#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 0 \
#       --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 \
#       --model_name interval_mlp400 --agent_type interval  --agent_name IntervalNet \
#       --lr 0.001 | tee ${OUTDIR}/Adam.log

# Incremental class
python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam \
       --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 10 \
       --batch_size 128 --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --lr 0.001 --kappa_epoch 10 --eps_epoch 10 --eps_max 0.1 | tee ${OUTDIR}/Adam_exp1.log

python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam \
       --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 20 \
       --batch_size 128 --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --lr 0.001 --kappa_epoch 10 --eps_epoch 20 --eps_max 0.1 | tee ${OUTDIR}/Adam_exp2.log

python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam \
       --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 10 \
       --batch_size 128 --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --lr 0.01 --kappa_epoch 10 --eps_epoch 10 --eps_max 0.1 | tee ${OUTDIR}/Adam_exp3.log

python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam \
       --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 20 \
       --batch_size 128 --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --lr 0.01 --kappa_epoch 10 --eps_epoch 20 --eps_max 0.1 | tee ${OUTDIR}/Adam_exp4.log


python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam \
       --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 30 \
       --batch_size 128 --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --lr 0.001 --kappa_epoch 10 --eps_epoch 10 --eps_max 0.1 | tee ${OUTDIR}/Adam_exp5.log

python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam \
       --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 30 \
       --batch_size 128 --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --lr 0.01 --kappa_epoch 10 --eps_epoch 10 --eps_max 0.1 | tee ${OUTDIR}/Adam_exp6.log
