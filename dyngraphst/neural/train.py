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
from torch.nn.utils import clip_grad_norm_
from .utils import make_schedule
from time import time

from typing import Callable


def load_data(path: str) -> tuple[BatchItems, BatchItems, BatchItems]:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_loaders(data: tuple[BatchItems, BatchItems, BatchItems],
                 device: str,
                 pad_token_id: int,
                 max_seq_len: int,
                 batch_size_train: int = 16,
                 batch_size_dev: int = 64,
                 cls_dist: int = -999) -> tuple[DataLoader, DataLoader, DataLoader]:
    train, dev, test = [[sample for sample in subset
                         if len(sample[0][0]) <= max_seq_len]
                         # and max(map(len, sample[1])) < max_type_depth]
                        for subset in data]
    collate_fn = make_collator(device, pad_token_id=pad_token_id, cls_dist=cls_dist)
    return (DataLoader(train, batch_size_train, shuffle=True, collate_fn=collate_fn),
            DataLoader(sorted(dev, key=lambda x: len(x[0][0])), batch_size_dev, shuffle=False, collate_fn=collate_fn),
            DataLoader(sorted(test, key=lambda x: len(x[0][0])), batch_size_dev, shuffle=False, collate_fn=collate_fn))


def train(device: str,
          encoder_core: str,
          bert_type: str,
          storage_dir: str,
          log_path: str,
          init_epoch: int,
          data_path: str,
          num_classes: int,
          max_dist: int,
          schedule_epochs: int,
          max_seq_len: int,
          depth_per_epoch: Callable[[int], int],
          pad_token_id: int,
          sep_token_id: int,
          num_epochs: int = None,
          batch_size: int = 16):

    def logprint(msg: str):
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
            print(msg)

    data = load_data(data_path)
    train_dl = make_loaders(data=data, device=device, max_seq_len=max_seq_len,
                            pad_token_id=pad_token_id, batch_size_train=batch_size)[0]
    model = Tagger(num_classes=num_classes,
                   max_dist=max_dist,
                   encoder_core=encoder_core,
                   bert_type=bert_type,
                   sep_token_id=sep_token_id).to(device)
    loss_fn = GroupedLoss(reduction='sum', label_smoothing=0.1)
    opt = AdamW([
        {'params': model.encoder.core.parameters(), 'lr': 1e-5},
        {'params': sum(map(list, (model.decoder.parameters(),
                                  model.dist_embedding.parameters(),
                                  model.embedder.parameters(),
                                  model.path_encoder.parameters(),
                                  model.encoder.aggregator.parameters(),)), []), 'lr': 1e-4}],
        weight_decay=1e-2)
    schedule = make_schedule(warmup_steps=int(0.1 * len(train_dl) * schedule_epochs),
                             warmdown_steps=int(0.9 * schedule_epochs * len(train_dl)),
                             total_steps=schedule_epochs * len(train_dl),
                             max_lr=1,
                             min_lr=1e-3)

    if init_epoch != 0:
        model.load_state_dict(torch.load(f'{storage_dir}/model_{init_epoch - 1}.pt'))
        opt.load_state_dict(torch.load(f'{storage_dir}/opt_{init_epoch - 1}.pt'))
    else:
        message = f'Training {encoder_core} for {schedule_epochs} epochs' \
                  f' with a max sequence length of {max_seq_len} and a maximum embedding distance of {max_dist}.\n'
        message += model.imprint
        logprint(message)

    scheduler = LambdaLR(opt,
                         [schedule for _ in range(len(opt.param_groups))],
                         last_epoch=-1 if init_epoch == 0 else len(train_dl) * init_epoch)

    for epoch in range(init_epoch, schedule_epochs if num_epochs is None else num_epochs):
        max_depth = depth_per_epoch(epoch)
        model.train()
        epoch_loss = 0.
        epoch_stats = defaultdict(lambda: (0, 0))
        now = time()
        for batch in train_dl:
            (token_ids, atn_mask, token_clusters,
             root_edge_index, root_edge_dist,
             root_to_node_index, node_ids,
             node_pos, numels) = batch
            opt.zero_grad(set_to_none=True)
            out = model.forward_train(input_ids=token_ids,
                                      attention_mask=atn_mask,
                                      token_clusters=token_clusters,
                                      node_to_root_index=root_to_node_index[:max_depth],
                                      node_ids=node_ids[:max_depth],
                                      node_pos=node_pos[:max_depth],
                                      root_dist=root_edge_dist,
                                      root_edge_index=root_edge_index)
            loss = loss_fn.forward_many(out, node_ids[:max_depth], numels[:max_depth]) / token_clusters.max()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.)
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
        correct, total = sum(map(lambda x: x[0], epoch_stats.values())), sum(map(lambda x: x[1], epoch_stats.values()))
        message += f'Total: {correct}/{total} ({correct / total:.2f})\n'
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
             max_depth: int,
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
    dl = make_loaders(data, device, pad_token_id=pad_token_id, max_seq_len=max_seq_len)[2 if test_set else 1]
    model.path_encoder.precompute(2 ** max_depth + 1)
    with torch.no_grad():
        dev_outs = []
        for batch in dl:
            (token_ids, atn_mask, token_clusters, root_edge_index,
             root_edge_dist, _, node_ids, _, _) = batch
            out = model.forward_dev(input_ids=token_ids, attention_mask=atn_mask,
                                    token_clusters=token_clusters, root_edge_index=root_edge_index,
                                    root_dist=root_edge_dist,
                                    max_type_depth=max_depth,
                                    first_binary=first_binary)
            sent_lens = token_clusters.max(dim=1).values
            dev_outs.append(([o.tolist() for o in out], [n.tolist() for n in node_ids], sent_lens.tolist()))
    if storage_path is not None:
        with open(storage_path, 'wb') as f:
            pickle.dump(dev_outs, f)
    return dev_outs
