from train import app as train_app

CONFIG_NAME = 'denoise_rnn'
CONFIG = {
    'data.dataset.path': 'data/ETTh1.csv',
    'data.loader': 'etth',
    'model.seq_len': 720,
    'model.pred_len': 96,
    'model.n_channels': 7,
    'model.d_model': 512,
    'model.dropout': 0.5,
    'model.patch_len': 48,
    'lr.init': 0.001,
    'lr.decay': 0.9,
    'data.batch_size': 256
}
GRID = {
    'lr.init': [0.001, 0.0006, 0.0003],
    'model.d_model': [512, 256, 128],
    'model.dropout': [0.5, 0.2, 0.1],
    'model.patch_len': [24, 48, 8],
}

best_config = {}
best_stat = {}


def write_best_config():
    global best_config
    global best_stat
    with open('best_config.txt', 'a') as f:
        f.write(f"Best config: {best_config}\n")
        f.write(f"Best stat: {best_stat}\n")
        f.write("***************************************************************\n")


def format_train_config(config):
    config = [f"--{k}={v}" for k, v in config.items()]
    return [CONFIG_NAME] + config


def train_step(cur_conf, next_conf):
    global best_config
    global best_stat

    if len(next_conf) == 0:
        print(f"Current config: {cur_conf}")
        cur_pred_len = cur_conf['model.pred_len']
        train_config = format_train_config(cur_conf)
        try:
            trainer = train_app(train_config)
            cur_stat = trainer.log.get_stat('test', 'test/mse')
            cur_best_config = best_config.get(cur_pred_len, None)
            cur_best_stat = best_stat.get(cur_pred_len, None)
            if cur_best_config is None or cur_best_stat is None or cur_stat < cur_best_stat:
                best_config[cur_pred_len] = cur_conf
                best_stat[cur_pred_len] = cur_stat
                print(f"Best config: {best_config}")
                print(f"Best stat: {best_stat}")
                write_best_config()
        except Exception as e:
            print(e)
        return

    cur_k = list(next_conf.keys())[0]
    for v in next_conf[cur_k]:
        temp_conf = {**cur_conf, cur_k: v}
        filtered_next_conf = {k: v for k, v in next_conf.items() if k != cur_k}
        train_step(temp_conf, filtered_next_conf)


def app():
    train_step(CONFIG, GRID)
    print("Best config: ", best_config)
    print("Best stat: ", best_stat)
    write_best_config()


if __name__ == '__main__':
    app()
