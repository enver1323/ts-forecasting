config=seg_rnn
seq_len=720
# for pred_len in 96 192 336 720
for pred_len in 96
do
    python -u train.py \
        $config \
        --data.dataset.path=data/ETTh1.csv \
        --data.loader=etth \
        --model.seq_len $seq_len \
        --model.pred_len $pred_len \
        --model.patch_len 48 \
        --model.n_channels 7 \
        --model.d_model 512 \
        --model.dropout 0.5 \
        --n_epochs 30 \
        --patience 10 \
        --data.batch_size 256 \
        --lr.init 0.001
done
