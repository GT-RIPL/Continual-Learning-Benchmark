GPUID=$1
OUTDIR=outputs/split_CIFAR100_cnn_interval_id
REPEAT=1
mkdir -p $OUTDIR

#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
#       --reg_coef 10 \
#       --schedule 150 \
#       --kappa_epoch 150 \
#       --eps_epoch 150 \
#       --eps_val 2 1 0.5 0.25 0.12 0.1 0.1 0.1 0.1 0.1 --eps_max 0 \
#       | tee ${OUTDIR}/tn1_id.log


python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
       --reg_coef 10 \
       --schedule 200 \
       --kappa_epoch 150 \
       --eps_epoch 100 \
       --eps_val 3 --eps_max 0 \
       | tee ${OUTDIR}/tn1_id.log

