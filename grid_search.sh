for use_content in 1; do
for lr in 0.005; do
for lr_poincare in 0; do
for reg_l2 in 1e-5 5e-5; do
for embedding_dim in 30; do

python3 deepfm_rank.py \
     --lr $lr --lr_poincare $lr_poincare --use_content $use_content \
     --n_epochs 50 --dataset 'TX' --embedding_dim $embedding_dim \
     --batch_size 4096 --optimizer 'Adam' --reg_l2 $reg_l2

done
done
done
done
done
