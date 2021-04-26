GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_it_test
REPEAT=2
mkdir -p $OUTDIR


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 0.1 0.1 0.08 0.05 0.01 --eps_epoch 20 --eps_max 1 \
#       --kappa_epoch 10 --schedule 20 \
#       | tee ${OUTDIR}/tn1.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 0.1 0.1 0.08 0.05 0.05 --eps_epoch 20 --eps_max 1 \
#       --kappa_epoch 10 --schedule 20 \
#       | tee ${OUTDIR}/tn2.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 0.5 0.2 0.1 0.05 0.05 --eps_epoch 20 --eps_max 0 \
#       --kappa_epoch 10 --schedule 20 \
#       | tee ${OUTDIR}/tn3.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 0.5 0.2 0.1 0.1 0.1 --eps_epoch 20 --eps_max 0 \
#       --kappa_epoch 10 --schedule 20 \
#       | tee ${OUTDIR}/tn4.log


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 0.5 0.2 0.1 0.1 0.1 --eps_epoch 20 --eps_max 1 \
#       --kappa_epoch 10 --schedule 20 \
#       | tee ${OUTDIR}/tn2_test.log


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 1 0.3 0.2 0.2 0.1 --eps_epoch 40 --eps_max 1 0.3 0.2 0.2 0.1 \
#       --kappa_epoch 10 --schedule 50 \
#       | tee ${OUTDIR}/tn2_test.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 2 0.3 0.2 0.2 0.1 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 20 --schedule 50 \
#       | tee ${OUTDIR}/tn2_test1.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 2 0.7 0.3 0.2 0.1 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 20 --schedule 50 \
#       | tee ${OUTDIR}/tn2_test2.log
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 2 0.7 0.5 0.4 0.3 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 20 --schedule 50 \
#       | tee ${OUTDIR}/tn2_test3.log


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 2 1 0.3 0.2 0.1 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 20 --schedule 50 \
#       | tee ${OUTDIR}/tn2_test4.log
#
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 2.5 0.9 0.4 0.2 0.1 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 20 --schedule 50 \
#       | tee ${OUTDIR}/tn2_test5.log
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 3 0.9 0.4 0.2 0.1 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 20 --schedule 50 \
#       | tee ${OUTDIR}/tn2_test6.log
#

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 20 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 50 --schedule 50 \
#       | tee ${OUTDIR}/tn2_test7.log


python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 2 \
       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
       --eps_val 20 12 4 1.2 0.4 --eps_epoch 50 --eps_max 0 \
       --kappa_epoch 50 --schedule 50 \
       | tee ${OUTDIR}/tn2_test8.log

