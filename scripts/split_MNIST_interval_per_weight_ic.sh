GPUID=$1
OUTDIR=outputs/split_MNIST_interval_ic_per_weight
REPEAT=10
mkdir -p $OUTDIR

#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class \
#       --optimizer Adam --force_out_dim 10 --no_class_remap --first_split_size 2 \
#       --other_split_size 2 --model_name interval_mlp400 --agent_type interval \
#       --agent_name IntervalNet --schedule 12 --batch_size 100 --lr 0.001 \
#       --kappa_epoch 4 --eps_epoch 12 --eps_val 2 0.9 0.1 0.1 0.1 --eps_max 0 --clipping \
#       | tee ${OUTDIR}/in_pw_zeros.log

#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class \
#       --optimizer Adam --force_out_dim 10 --no_class_remap --first_split_size 2 \
#       --other_split_size 2 --model_name interval_mlp400 --agent_type interval \
#       --agent_name IntervalNet --schedule 12 --batch_size 100 --lr 0.001 --eps_per_model \
#       --kappa_epoch 4 --eps_epoch 12 --eps_val 6 3 3 3 3 --eps_max 0 --clipping \
#       | tee ${OUTDIR}/in_pw_zeros1.log

#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class \
#       --optimizer Adam --force_out_dim 10 --no_class_remap --first_split_size 2 \
#       --other_split_size 2 --model_name interval_mlp400 --agent_type interval \
#       --agent_name IntervalNet --schedule 12 --batch_size 100 --lr 0.001 --eps_per_model \
#       --kappa_epoch 1 --eps_epoch 12 --eps_val 7 3 3 3 3 --eps_max 0 --clipping \
#       | tee ${OUTDIR}/in_pw.log

#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class \
#       --optimizer Adam --force_out_dim 10 --no_class_remap --first_split_size 2 \
#       --other_split_size 2 --model_name interval_mlp400 --agent_type interval \
#       --agent_name IntervalNet --schedule 12 --batch_size 100 --lr 0.001 --eps_per_model \
#       --kappa_epoch 1 --eps_epoch 12 --eps_val 6 3 3 3 3 --eps_max 0 --clipping \
#       | tee ${OUTDIR}/in_pw1.log

# BEST 32%
#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class \
#       --optimizer Adam --force_out_dim 10 --no_class_remap --first_split_size 2 \
#       --other_split_size 2 --model_name interval_mlp400 --agent_type interval \
#       --agent_name IntervalNet --schedule 12 --batch_size 100 --lr 0.001 --eps_per_model \
#       --kappa_epoch 1 --eps_epoch 12 --eps_val 7 7 2 2 1 --eps_max 0 --clipping \
#       | tee ${OUTDIR}/in_pw2.log

python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class \
       --optimizer Adam --force_out_dim 10 --no_class_remap --first_split_size 2 \
       --other_split_size 2 --model_name interval_mlp400 --agent_type interval \
       --agent_name IntervalNet --schedule 12 --batch_size 100 --lr 0.001 --eps_per_model \
       --kappa_epoch 1 --eps_epoch 12 --eps_val 7 7 2 2 1 --eps_max 0 --clipping \
       | tee ${OUTDIR}/in_pw3.log

# --eps_val 5 3 3 3 3 acc 31%