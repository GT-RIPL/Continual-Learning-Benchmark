GPUID=$1
OUTDIR=outputs/split_MNIST_interval_id
REPEAT=10
mkdir -p $OUTDIR

# Incremental Domain
python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 \
       --first_split_size 2 --other_split_size 2 --schedule 8 --batch_size 128 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --lr 0.001 --kappa_epoch 4 --eps_epoch 8 --eps_val 0.1 --eps_max 0.1 | tee ${OUTDIR}/IN_Adam_01.log

python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 \
       --first_split_size 2 --other_split_size 2 --schedule 16 --batch_size 128 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --lr 0.001 --kappa_epoch 8 --eps_epoch 16 --eps_val 0.2 --eps_max 0.2 | tee ${OUTDIR}/IN_Adam_02.log

python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 \
       --first_split_size 2 --other_split_size 2 --schedule 24 --batch_size 128 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --lr 0.001 --kappa_epoch 10 --eps_epoch 24 --eps_val 0.3 --eps_max 0.3 | tee ${OUTDIR}/IN_Adam_03.log


python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 \
       --first_split_size 2 --other_split_size 2 --schedule 8 --batch_size 128 --clipping \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --lr 0.001 --kappa_epoch 4 --eps_epoch 8 --eps_val 0.1 --eps_max 0.1 | tee ${OUTDIR}/IN_Adam_01_clipping.log

python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 \
       --first_split_size 2 --other_split_size 2 --schedule 16 --batch_size 128 --clipping \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --lr 0.001 --kappa_epoch 8 --eps_epoch 16 --eps_val 0.2 --eps_max 0.2 | tee ${OUTDIR}/IN_Adam_02_clipping.log

python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 \
       --first_split_size 2 --other_split_size 2 --schedule 24 --batch_size 128 --clipping \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --lr 0.001 --kappa_epoch 10 --eps_epoch 24 --eps_val 0.3 --eps_max 0.3 | tee ${OUTDIR}/IN_Adam_03_clipping.log




#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 \
#       --first_split_size 2 --other_split_size 2 --schedule 10 --batch_size 128 --interval_epoch 5 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --lr 0.001 --kappa_epoch 10 --eps_epoch 10 --eps_max 0.1 | tee ${OUTDIR}/Adam_exp1.log
#
#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 \
#       --first_split_size 2 --other_split_size 2 --schedule 20 --batch_size 128 --interval_epoch 5 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --lr 0.001 --kappa_epoch 10 --eps_epoch 20 --eps_max 0.1 | tee ${OUTDIR}/Adam_exp2.log
#
#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 \
#       --first_split_size 2 --other_split_size 2 --schedule 10 --batch_size 128 --interval_epoch 5 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --lr 0.01 --kappa_epoch 10 --eps_epoch 10 --eps_max 0.1 | tee ${OUTDIR}/Adam_exp3.log
#
#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 \
#       --first_split_size 2 --other_split_size 2 --schedule 20 --batch_size 128 --interval_epoch 5 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --lr 0.01 --kappa_epoch 10 --eps_epoch 20 --eps_max 0.05 | tee ${OUTDIR}/Adam_exp4.log
#
#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 \
#       --first_split_size 2 --other_split_size 2 --schedule 30 --batch_size 128 --interval_epoch 5 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --lr 0.001 --kappa_epoch 10 --eps_epoch 10 --eps_max 0.1 | tee ${OUTDIR}/Adam_exp5.log
