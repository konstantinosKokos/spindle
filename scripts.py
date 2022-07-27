import torch
from dyngraphpn.neural.train import train, evaluate


NUM_EPOCHS = 25
MAX_DIST = 6
MAX_SEQ_LEN = 199
NUM_SYMBOLS = 80
FIRST_BINARY = 31
MAX_DEPTH = 15
BATCH_SIZE = 64


def train_aethel(device: torch.device, storage_dir: str, log_path: str, init_epoch: int = 0,
                 data_path: str = './interface/vectorized.p', num_symbols: int = NUM_SYMBOLS,
                 max_dist: int = MAX_DIST, schedule_epochs: int = NUM_EPOCHS, max_seq_len: int = MAX_SEQ_LEN, **kwargs):
    train(device=device, encoder_core='GroNLP/bert-base-dutch-cased', bert_type='bert', storage_dir=storage_dir,
          log_path=log_path, init_epoch=init_epoch, data_path=data_path, num_classes=num_symbols,
          max_dist=max_dist, schedule_epochs=schedule_epochs, max_seq_len=max_seq_len,
          depth_per_epoch=lambda _: MAX_DEPTH, pad_token_id=3, sep_token_id=2, batch_size=BATCH_SIZE, **kwargs)


def evaluate_aethel(device: torch.device, model_path: str, data_path: str = './interface/vectorized.p',
                    storage_path: str = './output.p', max_dist: int = MAX_DIST,
                    max_seq_len: int = MAX_SEQ_LEN, max_type_depth: int = 16, test_set: bool = False):
    evaluate(device=device, encoder_core='GroNLP/bert-base-dutch-cased', bert_type='bert', model_path=model_path,
             data_path=data_path, storage_path=storage_path, num_classes=80, max_dist=max_dist,
             max_seq_len=max_seq_len, pad_token_id=3, max_depth=max_type_depth,
             sep_token_id=2, first_binary=31, test_set=test_set)
