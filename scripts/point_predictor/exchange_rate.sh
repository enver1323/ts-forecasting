# pred_lens=(96 192 336 720)
pred_lens=(96)

config="point_predictor"

for pred_len in ${pred_lens[@]};
do
    python train.py \
        $config \
        --data.dataset.path=data/exchange_rate.csv \
        --model.n_channels=1 \
        --model.context_size=336 \
        --model.initial_size=96 \
        --model.window_size=168 \
        --model.hidden_size=72 \
        --model.kernel_size=25 \
        --lr.init=0.0001 \
        --data.batch_size=32
done