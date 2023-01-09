import torch

from dyngraphpn.data.tokenization import load_data
from dyngraphpn.neural.batching import make_loader
from dyngraphpn.neural.model import Parser
from dyngraphpn.neural.loss import TaggingLoss, LinkingLoss
from dyngraphpn.neural.utils import make_schedule

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from collections import defaultdict
from time import time

from random import random


NUM_EPOCHS = 30
SCHEDULE_EPOCHS = 30
MAX_DIST = 6
MAX_SEQ_LEN = 199
NUM_SYMBOLS = 81
FIRST_BINARY = 32
MAX_DEPTH = 15
BATCH_SIZE = 64


def train(device: torch.device = 'cuda',
          encoder_core: str = 'GroNLP/bert-base-dutch-cased',
          bert_type: str = 'bert',
          storage_dir: str = './log/',
          log_path: str = './log/log.txt',
          init_epoch: int = 0,
          data_path: str = './data/vectorized.p',
          num_classes: int = NUM_SYMBOLS,
          max_dist: int = MAX_DIST,
          schedule_epochs: int = SCHEDULE_EPOCHS,
          max_seq_len: int = MAX_SEQ_LEN,
          max_depth: int = MAX_DEPTH,
          pad_token_id: int = 3,
          sep_token_id: int = 2,
          num_epochs: int = NUM_EPOCHS,
          batch_size: int = BATCH_SIZE):

    def logprint(msg: str):
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
            print(msg)

    data = load_data(data_path)[0]
    dl = make_loader(data=data, device=device, max_seq_len=max_seq_len,
                     pad_token_id=pad_token_id, batch_size=batch_size, sort=False)
    model = Parser(num_classes=num_classes,
                   max_dist=max_dist,
                   encoder_config_or_name=encoder_core,
                   bert_type=bert_type,
                   sep_token_id=sep_token_id).to(device)
    tagging_loss_fn = TaggingLoss(reduction='sum')
    linking_loss_fn = LinkingLoss()
    opt = AdamW([
        {'params': model.encoder.core.parameters(), 'lr': 1e-5},
        {'params': sum(map(list, (model.decoder.parameters(),
                                  model.dist_embedding.parameters(),
                                  model.embedder.parameters(),
                                  model.path_encoder.parameters(),
                                  model.encoder.aggregator.parameters(),
                                  model.linker.parameters(),)), []), 'lr': 1e-3}],
        weight_decay=1e-2)
    schedule = make_schedule(warmup_steps=int(0.1 * len(dl) * schedule_epochs),
                             warmdown_steps=int(0.9 * schedule_epochs * len(dl)),
                             total_steps=schedule_epochs * len(dl),
                             max_lr=1,
                             min_lr=1e-4)

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
                         last_epoch=-1 if init_epoch == 0 else len(dl) * init_epoch)

    ####################################################################################################################
    # main loop
    ####################################################################################################################
    for epoch in range(init_epoch, schedule_epochs if num_epochs is None else num_epochs):
        model.train()
        epoch_loss = (0, 0)
        tagging_accuracy = defaultdict(lambda: (0, 0))
        linking_accuracy = defaultdict(lambda: (0, 0))
        start = time()
        ################################################################################################################
        # epoch loop
        ################################################################################################################
        for batch in dl:
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
            loss = tagging_loss + linking_loss * 0.1 if random() > 0.2 else tagging_loss
            loss.backward()
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
        message += '=' * 64 + '\n'
        message += f'Time: {duration} ({len(dl)/duration} batch/sec)\n'
        message += f'Last LRs: {scheduler.get_last_lr()}\n'
        message += f'Tagging Loss: {epoch_tagging_loss / len(dl)}\n'
        message += f'Linking Loss: {epoch_linking_loss / len(dl)}\n'
        for depth in sorted(tagging_accuracy.keys()):
            correct, total = tagging_accuracy[depth]
            message += f'\tDepth {depth}: {correct}/{total} ({correct / total:.2f})\n'
        correct = sum(map(lambda x: x[0], tagging_accuracy.values()))
        total = sum(map(lambda x: x[1], tagging_accuracy.values()))
        message += f'\tTotal: {correct}/{total} ({correct / total:.2f})\n'
        for num_candidates in sorted(linking_accuracy.keys()):
            correct, total = linking_accuracy[num_candidates]
            message += f'\t{num_candidates} candidates: {correct}/{total} ({correct / total:.2f})\n'
        correct = sum(map(lambda x: x[0], linking_accuracy.values()))
        total = sum(map(lambda x: x[1], linking_accuracy.values()))
        message += f'\tTotal: {correct}/{total} ({correct / total:.2f})\n'
        logprint(message)
