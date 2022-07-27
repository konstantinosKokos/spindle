import pdb

from .batching import TokenizedSample, make_collator
import pickle
from torch.utils.data import DataLoader
from .model import Parser
from .loss import TaggingLoss, LinkingLoss

import torch
from collections import defaultdict
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from .utils import make_schedule
from time import time

from typing import Callable


TokenizedSamples = list[TokenizedSample]
_Dataset = tuple[TokenizedSamples, TokenizedSamples, TokenizedSamples]


def load_data(path: str) -> tuple[TokenizedSamples, TokenizedSamples, TokenizedSamples]:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_loaders(data: _Dataset,
                 device: torch.device,
                 pad_token_id: int,
                 max_seq_len: int,
                 batch_size_train: int = 16,
                 batch_size_dev: int = 64,
                 cls_dist: int = -999) -> tuple[DataLoader, DataLoader, DataLoader]:
    train, dev, test = [[sample for sample in subset
                         if len(sample[0][0]) <= max_seq_len]
                        for subset in data]
    collate_fn = make_collator(device, pad_token_id=pad_token_id, cls_dist=cls_dist)
    return (DataLoader(train, batch_size_train, shuffle=True, collate_fn=collate_fn),
            DataLoader(sorted(dev, key=lambda x: len(x[0][0])), batch_size_dev, shuffle=False, collate_fn=collate_fn),
            DataLoader(sorted(test, key=lambda x: len(x[0][0])), batch_size_dev, shuffle=False, collate_fn=collate_fn))


def train(device: torch.device,
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
    model = Parser(num_classes=num_classes,
                   max_dist=max_dist,
                   encoder_config_or_name=encoder_core,
                   bert_type=bert_type,
                   sep_token_id=sep_token_id).to(device)
    tagging_loss_fn = TaggingLoss(reduction='sum', label_smoothing=0.1)
    linking_loss_fn = LinkingLoss()
    opt = AdamW([
        {'params': model.encoder.core.parameters(), 'lr': 1e-5, 'weight_decay': 1e-2},
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

    ####################################################################################################################
    # main loop
    ####################################################################################################################
    for epoch in range(init_epoch, schedule_epochs if num_epochs is None else num_epochs):
        model.train()
        max_depth = depth_per_epoch(epoch)
        epoch_loss = (0, 0)
        tagging_accuracy = defaultdict(lambda: (0, 0))
        linking_accuracy = defaultdict(lambda: (0, 0))
        start = time()
        ################################################################################################################
        # epoch loop
        ################################################################################################################
        for batch in train_dl:
            opt.zero_grad(set_to_none=True)
            token_preds, matches = model.forward_train(
                input_ids=batch.encoder_batch.token_ids,
                attention_mask=batch.encoder_batch.atn_mask,
                token_clusters=batch.encoder_batch.cluster_ids,
                node_to_root_index=batch.decoder_batch.edge_index[:max_depth],
                node_ids=batch.decoder_batch.token_ids[:max_depth],
                node_pos=batch.decoder_batch.pos_ids[:max_depth],
                root_edge_index=batch.encoder_batch.edge_index,
                root_dist=batch.encoder_batch.edge_attr,
                link_indices=batch.parser_batch.indices)
            ############################################################################################################
            # backprop
            ############################################################################################################
            tagging_loss = tagging_loss_fn(token_preds, batch.decoder_batch.token_ids[:max_depth])
            tagging_loss = tagging_loss / batch.encoder_batch.cluster_ids.max()
            linking_loss = linking_loss_fn(matches)
            loss = tagging_loss + linking_loss * 0.1
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            scheduler.step()
            ############################################################################################################
            # logging
            ############################################################################################################
            prev_tagging_loss, prev_linking_loss = epoch_loss
            epoch_loss = (prev_tagging_loss + tagging_loss.item(), prev_linking_loss + linking_loss.item())
            with torch.no_grad():
                for depth, depth_preds in enumerate(token_preds):
                    prev_correct, prev_total = tagging_accuracy[depth]
                    batch_correct = depth_preds.argmax(dim=1).eq(batch.decoder_batch.token_ids[depth]).sum().item()
                    correct = prev_correct + batch_correct
                    total = prev_total + len(batch.decoder_batch.token_ids[depth])
                    tagging_accuracy[depth] = (correct, total)
                for match in matches:
                    batch_size, num_candidates = match.shape[:-1]
                    prev_correct, prev_total = linking_accuracy[num_candidates]
                    truth = torch.arange(num_candidates, device=device).repeat(batch_size)
                    batch_correct = match.argmax(dim=2).flatten().eq(truth).sum().item()
                    correct = prev_correct + batch_correct
                    total = prev_total + batch_size * num_candidates
                    linking_accuracy[num_candidates] = (correct, total)
        ################################################################################################################
        # epoch report
        ################################################################################################################
        duration = time() - start
        torch.save(model.state_dict(), f'{storage_dir}/model_{epoch}.pt')
        torch.save(opt.state_dict(), f'{storage_dir}/opt_{epoch}.pt')
        epoch_tagging_loss, epoch_linking_loss = epoch_loss
        message = f'Epoch {epoch}\n'
        message += '=' * 20 + '\n'
        message += f'Time: {duration} ({len(train_dl)/duration} batch/sec)\n'
        message += f'Last LRs: {scheduler.get_last_lr()}\n'
        message += f'Tagging Loss: {epoch_tagging_loss / len(train_dl)}\n'
        message += f'Linking Loss: {epoch_linking_loss / len(train_dl)}\n'
        for depth in sorted(tagging_accuracy.keys()):
            correct, total = tagging_accuracy[depth]
            message += f'Depth {depth}: {correct}/{total} ({correct / total:.2f})\n'
        correct = sum(map(lambda x: x[0], tagging_accuracy.values()))
        total = sum(map(lambda x: x[1], tagging_accuracy.values()))
        message += f'Total: {correct}/{total} ({correct / total:.2f})\n'
        for num_candidates in sorted(linking_accuracy.keys()):
            correct, total = linking_accuracy[num_candidates]
            message += f'{num_candidates} candidates: {correct}/{total} ({correct / total:.2f})\n'
        correct = sum(map(lambda x: x[0], linking_accuracy.values()))
        total = sum(map(lambda x: x[1], linking_accuracy.values()))
        message += f'Total: {correct}/{total} ({correct / total:.2f})\n'
        logprint(message)


# todo
def evaluate(device: torch.device,
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
             test_set: bool,
             batch_size: int = 64):
    model = Parser(num_classes=num_classes,
                   max_dist=max_dist,
                   encoder_config_or_name=encoder_core,
                   bert_type=bert_type,
                   sep_token_id=sep_token_id).to(device)

    model.load(model_path, map_location=device)
    model.eval()
    data = load_data(data_path)
    dls = make_loaders(data, device, pad_token_id=pad_token_id, max_seq_len=max_seq_len, batch_size_dev=batch_size)
    dl = dls[2 if test_set else 1]
    model.path_encoder.precompute(2 ** max_depth + 1)
    start = time()
    with torch.no_grad():
        dev_outs = []
        for batch in dl:
            preds, _ , _ = model.forward_dev(input_ids=batch.encoder_batch.token_ids,
                                             attention_mask=batch.encoder_batch.atn_mask,
                                             token_clusters=batch.encoder_batch.cluster_ids,
                                             root_edge_index=batch.encoder_batch.edge_index,
                                             root_dist=batch.encoder_batch.edge_attr,
                                             max_type_depth=max_depth,
                                             first_binary=first_binary)
            sent_lens = batch.encoder_batch.cluster_ids.max(dim=1).values
            dev_outs.append(([o.tolist() for o in preds],
                             [n.tolist() for n in batch.decoder_batch.token_ids],
                             sent_lens.tolist()))
    end = time()
    print(end - start)
    if storage_path is not None:
        with open(storage_path, 'wb') as f:
            pickle.dump(dev_outs, f)
    return dev_outs
