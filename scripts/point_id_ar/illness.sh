pred_lens=(24 36 48 60)
# pred_lens=(24 )

# config="point_id"

# for pred_len in ${pred_lens[@]};
# do
#     python train.py \
#         $config \
#         --data.dataset.path=data/national_illness.csv \
#         --model.n_channels=7 \
#         --model.seq_len=336 \
#         --model.pred_len=$pred_len \
#         --model.n_points=12 \
#         --data.batch_size=32
# done


config="point_id_ar"

for pred_len in ${pred_lens[@]};
do
    python train.py \
        $config \
        --data.dataset.path=data/national_illness.csv \
        --model.n_choices=4 \
        --model.n_channels=7 \
        --model.seq_len=104 \
        --model.pred_len=$pred_len \
        --data.batch_size=32
done