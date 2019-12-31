num_folds=$1; shift;
cross_validation_dir=$1; shift;
save_dir=$1; shift;
max_epoch=$1; shift;
eval_step=$1; shift;
train_batch=$1; shift;
test_batch=$1; shift;
lr=$1; shift;

for i in $(seq 1 ${num_folds})
do
  fold=`expr $i - 1`

  exec_str="python main_reid.py train \
    --train-set-path=$cross_validation_dir/$fold/train.txt \
    --gallery-set-path=$cross_validation_dir/$fold/gallery.txt \
    --query-set-path=$cross_validation_dir/$fold/query.txt \
    --save_dir=$save_dir/$fold \
    --max_epoch=$max_epoch \
    --eval_step=$eval_step \
    --dataset=retail \
    --lr=$lr \
    --test_batch=$test_batch \
    --train_batch=$train_batch \
    --optim=adam \
    --adjust_lr \
  "
  echo "Training on Fold $fold"
  echo $exec_str
  $($exec_str)
done