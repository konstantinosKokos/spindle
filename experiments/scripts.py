from dyngraphst.neural.train import train, evaluate


def train_nl(device: str, storage_dir: str, log_path: str, init_epoch: int = 0,
             data_path: str = './experiments/aethel/vectorized.p', group_size: int = 32,
             num_symbols: int = 61, max_dist: int = 5, num_epochs: int = 40, max_seq_len: int = 50,
             max_type_depth: int = 10):
    return train(device=device, encoder='./GroNLP/bert-base-dutch-cased', storage_dir=storage_dir, log_path=log_path,
                 init_epoch=init_epoch, data_path=data_path, group_size=group_size, num_symbols=num_symbols,
                 max_dist=max_dist, num_epochs=num_epochs, max_seq_len=max_seq_len, max_type_depth=max_type_depth,
                 pad_token_id=3, sep_token_id=2,)


def train_en(device: str, storage_dir: str, log_path: str, init_epoch: int = 0,
             data_path: str = './experiments/ccg/vectorized.p', group_size: int = 39,
             num_symbols: int = 42, max_dist: int = 5, num_epochs: int = 40, max_seq_len: int = 50,
             max_type_depth: int = 10):
    return train(device=device, encoder='bert-base-cased', storage_dir=storage_dir, log_path=log_path,
                 init_epoch=init_epoch, data_path=data_path, group_size=group_size, num_symbols=num_symbols,
                 max_dist=max_dist, num_epochs=num_epochs, max_seq_len=max_seq_len, max_type_depth=max_type_depth,
                 pad_token_id=0, sep_token_id=102)


def evaluate_en(device: str, model_path: str, data_path: str = './experiments/ccg/vectorized.p',
                storage_path: str = None, num_classes: int = 42, max_dist: int = 5,
                max_seq_len: int = 50, max_type_depth: int = 10):
    return evaluate(device=device, encoder='./GroNLP/bert-base-dutch-cased', model_path=model_path,
                    data_path=data_path, storage_path=storage_path, num_classes=num_classes, max_dist=max_dist,
                    max_seq_len=max_seq_len, pad_token_id=0, max_type_depth=max_type_depth,
                    sep_token_id=102)


def evaluate_nl(device: str, model_path: str, data_path: str = './experiments/aethel/vectorized.p',
                storage_path: str = None, num_classes: int = 61, max_dist: int = 5,
                max_seq_len: int = 50, max_type_depth: int = 10):
    return evaluate(device=device, encoder='./GroNLP/bert-base-dutch-cased', model_path=model_path,
                    data_path=data_path, storage_path=storage_path, num_classes=num_classes, max_dist=max_dist,
                    max_seq_len=max_seq_len, pad_token_id=3, max_type_depth=max_type_depth,
                    sep_token_id=2)
