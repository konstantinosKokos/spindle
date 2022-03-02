from dyngraphst.neural.train import train, evaluate

# default hyperparameters across datasets
NUM_EPOCHS = 12
MAX_DIST = 8


def train_ccgbank(device: str, storage_dir: str, log_path: str, init_epoch: int = 0,
                  data_path: str = './experiments/ccg/vectorized_original.p',
                  max_dist: int = MAX_DIST, schedule_epochs: int = NUM_EPOCHS, max_seq_len: int = 199, **kwargs):
    train(device=device, encoder_core='roberta-base', bert_type='roberta', storage_dir=storage_dir,
          log_path=log_path, init_epoch=init_epoch, data_path=data_path, num_classes=37,
          max_dist=max_dist, schedule_epochs=schedule_epochs, max_seq_len=max_seq_len,
          depth_per_epoch=lambda _: 7, pad_token_id=1, sep_token_id=2, **kwargs)


def evaluate_ccgbank(device: str, model_path: str, data_path: str = './experiments/ccg/vectorized_original.p',
                     storage_path: str = None, max_dist: int = MAX_DIST,
                     max_seq_len: int = 199, max_type_depth: int = 10, test_set: bool = False):
    evaluate(device=device, encoder_core='roberta-base', bert_type='roberta', model_path=model_path,
             data_path=data_path, storage_path=storage_path, num_classes=37, max_dist=max_dist,
             max_seq_len=max_seq_len, pad_token_id=1, max_depth=max_type_depth,
             sep_token_id=2, first_binary=35, test_set=test_set)



# def train_nl(device: str, storage_dir: str, log_path: str, init_epoch: int = 0,
#              data_path: str = './experiments/aethel/vectorized.p', num_symbols: int = 60,
#              max_dist: int = 7, num_epochs: int = 20, max_seq_len: int = 170,
#              max_type_depth: int = 12):
#     return train(device=device, encoder='GroNLP/bert-base-dutch-cased', storage_dir=storage_dir, log_path=log_path,
#                  init_epoch=init_epoch, data_path=data_path, num_symbols=num_symbols,
#                  max_dist=max_dist, num_epochs=num_epochs, max_seq_len=max_seq_len, max_type_depth=max_type_depth,
#                  pad_token_id=3, sep_token_id=2,)
#
#
# def evaluate_nl(device: str, model_path: str, data_path: str = './experiments/aethel/vectorized.p',
#                 storage_path: str = None,  max_dist: int = 5,
#                 max_seq_len: int = 199, max_type_depth: int = 10, test_set: bool = False):
#     return evaluate(device=device, encoder='GroNLP/bert-base-dutch-cased', model_path=model_path,
#                     data_path=data_path, storage_path=storage_path, num_classes=61, max_dist=max_dist,
#                     max_seq_len=max_seq_len, pad_token_id=3, max_type_depth=max_type_depth,
#                     sep_token_id=2, first_binary=31, test_set=test_set)


#
# def train_nl_sparse(device: str, storage_dir: str, log_path: str, init_epoch: int = 0,
#                     data_path: str = './experiments/aethel/vectorized_sparse.p', num_symbols: int = 5299,
#                     max_dist: int = 7, num_epochs: int = 12, max_seq_len: int = 150,
#                     max_type_depth: int = 12):
#     return train(device=device, encoder='GroNLP/bert-base-dutch-cased', storage_dir=storage_dir, log_path=log_path,
#                  init_epoch=init_epoch, data_path=data_path, num_symbols=num_symbols,
#                  max_dist=max_dist, num_epochs=num_epochs, max_seq_len=max_seq_len, max_type_depth=max_type_depth,
#                  pad_token_id=3, sep_token_id=2,)
#
#
# def train_rebank(device: str, storage_dir: str, log_path: str, init_epoch: int = 0,
#                  data_path: str = './experiments/ccg/vectorized.p', num_symbols: int = 41,
#                  max_dist: int = 12, num_epochs: int = 12, max_seq_len: int = 170,
#                  max_type_depth: int = 10):
#     return train(device=device, encoder='roberta-base', storage_dir=storage_dir, log_path=log_path,
#                  init_epoch=init_epoch, data_path=data_path, num_symbols=num_symbols,
# #                  max_dist=max_dist, num_epochs=num_epochs, max_seq_len=max_seq_len, max_type_depth=max_type_depth,
# #                  pad_token_id=1, sep_token_id=2)
#
#
# def evaluate_rebank(device: str, model_path: str, data_path: str = './experiments/ccg/vectorized.p',
#                     storage_path: str = None, max_dist: int = 12,
#                     max_seq_len: int = 199, max_type_depth: int = 10, test_set: bool = False):
#     return evaluate(device=device, encoder='roberta-base', model_path=model_path,
#                     data_path=data_path, storage_path=storage_path, num_classes=41, max_dist=max_dist,
#                     max_seq_len=max_seq_len, pad_token_id=1, max_type_depth=max_type_depth,
#                     sep_token_id=2, first_binary=39, test_set=test_set)


