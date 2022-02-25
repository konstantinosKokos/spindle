import pdb

from .batching import BatchItems, make_collator
import pickle
from torch.utils.data import DataLoader
from .model import Tagger
from .loss import GroupedLoss

import torch
from collections import defaultdict
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from .utils import make_schedule
from time import time


def load_data(path: str) -> tuple[BatchItems, BatchItems, BatchItems]:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_loaders(data: tuple[BatchItems, BatchItems, BatchItems],
                 device: str,
                 pad_token_id: int,
                 max_seq_len: int,
                 batch_size_train: int = 16,
                 batch_size_dev: int = 256,
                 cls_dist: int = -999,
                 ) -> tuple[DataLoader, DataLoader, DataLoader]:
    train, dev, test = [[sample for sample in subset if len(sample[0][0]) <= max_seq_len] for subset in data]
    collate_fn = make_collator(device, pad_token_id=pad_token_id, cls_dist=cls_dist)
    return (DataLoader(train, batch_size_train, shuffle=True, collate_fn=collate_fn),
            DataLoader(dev, batch_size_dev, shuffle=False, collate_fn=collate_fn),
            DataLoader(test, batch_size_dev, shuffle=False, collate_fn=collate_fn))


def train(device: str,
          encoder_core: str,
          bert_type: str,
          storage_dir: str,
          log_path: str,
          init_epoch: int,
          data_path: str,
          num_classes: int,
          max_dist: int,
          num_epochs: int,
          max_seq_len: int,
          max_type_depth: int,
          pad_token_id: int,
          sep_token_id: int):

    def logprint(msg: str):
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
            print(msg)

    data = load_data(data_path)
    train_dl, val_dl, test_dl = make_loaders(data, device, max_seq_len=max_seq_len, pad_token_id=pad_token_id)
    model = Tagger(num_classes=num_classes,
                   max_dist=max_dist,
                   encoder_core=encoder_core,
                   bert_type=bert_type,
                   sep_token_id=sep_token_id).to(device)
    loss_fn = GroupedLoss(reduction='sum')
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    schedule = make_schedule(warmup_steps=int(0.75 * len(train_dl)),
                             warmdown_steps=int((num_epochs - 0.25) * len(train_dl)),
                             total_steps=num_epochs * len(train_dl),
                             max_lr=1,
                             min_lr=1e-2)

    if init_epoch != 0:
        model.load_state_dict(torch.load(f'{storage_dir}/model_{init_epoch - 1}.pt'))
        opt.load_state_dict(torch.load(f'{storage_dir}/opt_{init_epoch - 1}.pt'))
    else:
        message = f'Training {encoder_core} for {num_epochs} epochs' \
                  f' with a max sequence length of {max_seq_len} and a maximum embedding distance of {max_dist}'
        logprint(message)

    scheduler = LambdaLR(opt,
                         [schedule for _ in range(len(opt.param_groups))],
                         last_epoch=-1 if init_epoch == 0 else len(train_dl) * init_epoch)

    for epoch in range(init_epoch, num_epochs):
        model.train()
        epoch_loss = 0.
        epoch_stats = defaultdict(lambda: (0, 0))
        now = time()
        for batch in train_dl:
            (token_ids, atn_mask, token_clusters, root_edge_index,
             root_edge_dist, root_to_node_index, node_ids, node_pos, numels) = batch
            opt.zero_grad(set_to_none=True)
            out = model.forward_train(input_ids=token_ids,
                                      attention_mask=atn_mask,
                                      token_clusters=token_clusters,
                                      node_to_root_index=root_to_node_index[:max_type_depth],
                                      node_ids=node_ids[:max_type_depth],
                                      node_pos=node_pos[:max_type_depth],
                                      root_to_root_dist=root_edge_dist,
                                      root_to_root_index=root_edge_index)
            loss = loss_fn.forward_many(out, node_ids[:max_type_depth], numels[:max_type_depth]) / token_clusters.max()
            loss.backward()
            opt.step()
            scheduler.step()
            epoch_loss += loss.item()
            for depth in range(len(out)):
                correct = epoch_stats[depth][0] + out[depth].argmax(dim=1).eq(node_ids[depth]).sum().item()
                total = epoch_stats[depth][1] + len(node_ids[depth])
                epoch_stats[depth] = (correct, total)
        duration = time() - now
        torch.save(model.state_dict(), f'{storage_dir}/model_{epoch}.pt')
        torch.save(opt.state_dict(), f'{storage_dir}/opt_{epoch}.pt')
        message = f'Epoch {epoch}\n'
        message += '=' * 20 + '\n'
        message += f'Time: {duration} ({len(train_dl)/duration} batch/sec)\n'
        message += f'Last LRs: {scheduler.get_last_lr()}\n'
        message += f'Loss: {epoch_loss / len(train_dl)}\n'
        for depth in sorted(epoch_stats.keys()):
            correct, total = epoch_stats[depth]
            message += f'Depth {depth}: {correct}/{total} ({correct / total:.2f})\n'
        logprint(message)


def evaluate(device: str,
             encoder_core: str,
             bert_type: str,
             model_path: str,
             data_path: str,
             storage_path: str,
             num_classes: int,
             max_dist: int,
             max_seq_len: int,
             pad_token_id: int,
             max_type_depth: int,
             sep_token_id: int,
             first_binary: int,
             test_set: bool):
    model = Tagger(num_classes=num_classes,
                   max_dist=max_dist,
                   encoder_core=encoder_core,
                   bert_type=bert_type,
                   sep_token_id=sep_token_id).to(device)
    model.load(model_path, map_location=device)
    model.eval()
    data = load_data(data_path)
    dl = make_loaders(data, device, max_seq_len=max_seq_len, pad_token_id=pad_token_id)[2 if test_set else 1]
    model.path_encoder.precompute(2 ** max_type_depth + 1)
    with torch.no_grad():
        dev_outs = []
        for batch in dl:
            (token_ids, atn_mask, token_clusters, root_edge_index,
             root_edge_dist, _, node_ids, _, _) = batch
            out = model.forward_dev(input_ids=token_ids, attention_mask=atn_mask,
                                    token_clusters=token_clusters, root_to_root_index=root_edge_index,
                                    root_to_root_dist=root_edge_dist,
                                    max_type_depth=max_type_depth,
                                    first_binary=first_binary)
            sent_lens = token_clusters.max(dim=1).values
            dev_outs.append(([o.tolist() for o in out], [n.tolist() for n in node_ids], sent_lens.tolist()))
    if storage_path is not None:
        with open(storage_path, 'wb') as f:
            pickle.dump(dev_outs, f)
    return dev_outs
