from dyngraphst.neural.train import train, evaluate

# default hyperparameters across datasets
NUM_EPOCHS = 25
MAX_DIST = 6
MAX_SEQ_LEN = 199


def train_ccgbank(device: str, storage_dir: str, log_path: str, init_epoch: int = 0,
                  data_path: str = './experiments/ccg/vectorized_original.p',
                  max_dist: int = MAX_DIST, schedule_epochs: int = NUM_EPOCHS, max_seq_len: int = MAX_SEQ_LEN,
                  **kwargs):
    train(device=device, encoder_core='roberta-base', bert_type='roberta', storage_dir=storage_dir,
          log_path=log_path, init_epoch=init_epoch, data_path=data_path, num_classes=37,
          max_dist=max_dist, schedule_epochs=schedule_epochs, max_seq_len=max_seq_len,
          depth_per_epoch=lambda _: 7, pad_token_id=1, sep_token_id=2, **kwargs)


def evaluate_ccgbank(device: str, model_path: str, data_path: str = './experiments/ccg/vectorized_original.p',
                     storage_path: str = None, max_dist: int = MAX_DIST,
                     max_seq_len: int = MAX_SEQ_LEN, max_type_depth: int = 10, test_set: bool = False):
    evaluate(device=device, encoder_core='roberta-base', bert_type='roberta', model_path=model_path,
             data_path=data_path, storage_path=storage_path, num_classes=37, max_dist=max_dist,
             max_seq_len=max_seq_len, pad_token_id=1, max_depth=max_type_depth,
             sep_token_id=2, first_binary=35, test_set=test_set)


def train_aethel(device: str, storage_dir: str, log_path: str, init_epoch: int = 0,
                 data_path: str = './experiments/aethel/vectorized.p', num_symbols: int = 60,
                 max_dist: int = MAX_DIST, schedule_epochs: int = NUM_EPOCHS, max_seq_len: int = MAX_SEQ_LEN, **kwargs):
    train(device=device, encoder_core='GroNLP/bert-base-dutch-cased', bert_type='bert', storage_dir=storage_dir,
          log_path=log_path, init_epoch=init_epoch, data_path=data_path, num_classes=num_symbols,
          max_dist=max_dist, schedule_epochs=schedule_epochs, max_seq_len=max_seq_len,
          depth_per_epoch=lambda _: 11, pad_token_id=3, sep_token_id=2, **kwargs)


def evaluate_aethel(device: str, model_path: str, data_path: str = './experiments/aethel/vectorized.p',
                    storage_path: str = None, max_dist: int = MAX_DIST,
                    max_seq_len: int = MAX_SEQ_LEN, max_type_depth: int = 12, test_set: bool = False):
    evaluate(device=device, encoder_core='GroNLP/bert-base-dutch-cased', bert_type='bert', model_path=model_path,
             data_path=data_path, storage_path=storage_path, num_classes=60, max_dist=max_dist,
             max_seq_len=max_seq_len, pad_token_id=3, max_depth=max_type_depth,
             sep_token_id=2, first_binary=31, test_set=test_set)


def train_nl_sparse(device: str, storage_dir: str, log_path: str, init_epoch: int = 0,
                    data_path: str = './experiments/aethel/vectorized_sparse.p', num_symbols: int = 5298,
                    max_dist: int = MAX_DIST, schedule_epochs: int = NUM_EPOCHS, max_seq_len: int = MAX_SEQ_LEN,
                    **kwargs):
    train(device=device, encoder_core='GroNLP/bert-base-dutch-cased', bert_type='bert', storage_dir=storage_dir,
          log_path=log_path, init_epoch=init_epoch, data_path=data_path, num_classes=num_symbols,
          max_dist=max_dist, schedule_epochs=schedule_epochs, max_seq_len=max_seq_len,
          depth_per_epoch=lambda _: 2, pad_token_id=3, sep_token_id=2, **kwargs)


def evaluate_nl_sparse(device: str, model_path: str, data_path: str = './experiments/aethel/vectorized_sparse.p',
                       storage_path: str = None, max_dist: int = MAX_DIST, max_seq_len: int = MAX_SEQ_LEN,
                       max_type_depth: int = 2, test_set: bool = False):
    evaluate(device=device, encoder_core='GroNLP/bert-base-dutch-cased', bert_type='bert', model_path=model_path,
             data_path=data_path, storage_path=storage_path, num_classes=5298, max_dist=max_dist,
             max_seq_len=max_seq_len, pad_token_id=3, max_depth=max_type_depth,
             sep_token_id=2, first_binary=999999, test_set=test_set)


def train_rebank(device: str, storage_dir: str, log_path: str, init_epoch: int = 0,
                 data_path: str = './experiments/ccg/vectorized_rebank.p',
                 max_dist: int = MAX_DIST, schedule_epochs: int = NUM_EPOCHS, max_seq_len: int = MAX_SEQ_LEN, **kwargs):
    train(device=device, encoder_core='roberta-base', bert_type='roberta', storage_dir=storage_dir,
          log_path=log_path, init_epoch=init_epoch, data_path=data_path, num_classes=40,
          max_dist=max_dist, schedule_epochs=schedule_epochs, max_seq_len=max_seq_len,
          depth_per_epoch=lambda _: 7, pad_token_id=1, sep_token_id=2, **kwargs)


def evaluate_rebank(device: str, model_path: str, data_path: str = './experiments/ccg/vectorized_rebank.p',
                    storage_path: str = None, max_dist: int = MAX_DIST,
                    max_seq_len: int = MAX_SEQ_LEN, max_type_depth: int = 10, test_set: bool = False):
    evaluate(device=device, encoder_core='roberta-base', bert_type='roberta', model_path=model_path,
             data_path=data_path, storage_path=storage_path, num_classes=40, max_dist=max_dist,
             max_seq_len=max_seq_len, pad_token_id=1, max_depth=max_type_depth,
             sep_token_id=2, first_binary=38, test_set=test_set)


def train_french(device: str, storage_dir: str, log_path: str, init_epoch: int = 0,
                 data_path: str = './experiments/french_tlg/vectorized.p',
                 max_dist: int = MAX_DIST, schedule_epochs: int = NUM_EPOCHS, max_seq_len: int = MAX_SEQ_LEN, **kwargs):
    train(device=device, encoder_core='camembert-base', bert_type='camembert', storage_dir=storage_dir,
          log_path=log_path, init_epoch=init_epoch, data_path=data_path, num_classes=27,
          max_dist=max_dist, schedule_epochs=schedule_epochs, max_seq_len=max_seq_len,
          depth_per_epoch=lambda _: 8, pad_token_id=1, sep_token_id=6, **kwargs)


def evaluate_french(device: str, model_path: str, data_path: str = './experiments/french_tlg/vectorized.p',
                    storage_path: str = None, max_dist: int = MAX_DIST,
                    max_seq_len: int = MAX_SEQ_LEN, max_type_depth: int = 8, test_set: bool = False):
    evaluate(device=device, encoder_core='camembert-base', bert_type='camembert', model_path=model_path,
             data_path=data_path, storage_path=storage_path, num_classes=27, max_dist=max_dist,
             max_seq_len=max_seq_len, pad_token_id=1, max_depth=max_type_depth,
             sep_token_id=6, first_binary=19, test_set=test_set)


def train_french_sparse(device: str, storage_dir: str, log_path: str, init_epoch: int = 0,
                        data_path: str = './experiments/french_tlg/vectorized_sparse.p',
                        max_dist: int = MAX_DIST, schedule_epochs: int = NUM_EPOCHS, max_seq_len: int = MAX_SEQ_LEN,
                        **kwargs):
    train(device=device, encoder_core='camembert-base', bert_type='camembert', storage_dir=storage_dir,
          log_path=log_path, init_epoch=init_epoch, data_path=data_path, num_classes=852,
          max_dist=max_dist, schedule_epochs=schedule_epochs, max_seq_len=max_seq_len,
          depth_per_epoch=lambda _: 2, pad_token_id=1, sep_token_id=6, **kwargs)


def evaluate_french_sparse(device: str, model_path: str, data_path: str = './experiments/french_tlg/vectorized_sparse.p',
                           storage_path: str = None, max_dist: int = MAX_DIST,
                           max_seq_len: int = MAX_SEQ_LEN, max_type_depth: int = 2, test_set: bool = False):
    evaluate(device=device, encoder_core='camembert-base', bert_type='camembert', model_path=model_path,
             data_path=data_path, storage_path=storage_path, num_classes=852, max_dist=max_dist,
             max_seq_len=max_seq_len, pad_token_id=1, max_depth=max_type_depth,
             sep_token_id=6, first_binary=9999, test_set=test_set)
