GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_id_test
REPEAT=2
mkdir -p $OUTDIR


# 75%
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 2 0.5 0.5 0.5 0.5 --eps_epoch 40 20 20 20 20 --eps_max 0 \
#       --kappa_epoch 20 10 10 10 10 --schedule 40 20 20 20 20 \
#       | tee ${OUTDIR}/test.log


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 0.5 0.2 0.1 0.1 0.1 --eps_epoch 20 --eps_max 0 \
#       --kappa_epoch 10 --schedule 20 \
#       | tee ${OUTDIR}/tn1.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 0.1 0.1 0.08 0.05 0.05 --eps_epoch 20 --eps_max 0 \
#       --kappa_epoch 10 --schedule 20 \
#       | tee ${OUTDIR}/tn2.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 0.2 0.1 0.08 0.05 0.05 --eps_epoch 20 --eps_max 0 \
#       --kappa_epoch 10 --schedule 20 \
#       | tee ${OUTDIR}/tn3.log

python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
       --eps_val 1.5 0.5 0.2 0.1 0.05 --eps_epoch 50 --eps_max 0 \
       --kappa_epoch 15 --schedule 50 \
       | tee ${OUTDIR}/tn5.log


python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
       --eps_val 1.5 0.5 0.2 0.1 0.05 --eps_epoch 50 --eps_max 0 \
       --kappa_epoch 20 --schedule 50 \
       | tee ${OUTDIR}/tn55.log


python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
       --eps_val 2 0.8 0.3 0.2 0.1 --eps_epoch 50 --eps_max 0 \
       --kappa_epoch 20 --schedule 50 \
       | tee ${OUTDIR}/tn51.log


python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
       --eps_val 3 1 0.5 0.3 0.1 --eps_epoch 50 --eps_max 0 \
       --kappa_epoch 20 --schedule 50 \
       | tee ${OUTDIR}/tn52.log


python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
       --eps_val 1.8 0.9 0.45 0.22 0.11 --eps_epoch 50 --eps_max 0 \
       --kappa_epoch 20 --schedule 50 \
       | tee ${OUTDIR}/tn53.log