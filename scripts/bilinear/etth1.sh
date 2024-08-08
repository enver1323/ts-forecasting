config=bilinear
seq_len=336
# for pred_len in 96 192 336 720
for pred_len in 96
do
    python -u train.py \
        $config \
        --data.dataset.path=data/ETTh1.csv \
        --data.loader=etth \
        --model.seq_len $seq_len \
        --model.pred_len $pred_len \
        --model.patch_len 24 \
        --model.n_channels 7 \
        --model.d_model 64 \
        --model.dropout 0.2 \
        --n_epochs 30 \
        --patience 3 \
        --data.batch_size 64 \
        --lr.init 0.001 \
        --lr.decay 0.8
done
