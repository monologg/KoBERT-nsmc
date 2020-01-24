import os
import time
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from utils import set_seed, compute_metrics, get_label, MODEL_CLASSES

logger = logging.getLogger(__name__)


def _get_device_spec(device):
    ordinal = xm.get_ordinal(defval=-1)
    return str(device) if ordinal < 0 else '{}/{}'.format(device, ordinal)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

        self.bert_config = self.config_class.from_pretrained(args.model_name_or_path, num_labels=self.num_labels, finetuning_task=args.task)

    def _train_update(self, device, step_num, loss, tracker):
        print('[{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
            _get_device_spec(device), step_num, loss, tracker.rate(), tracker.global_rate(),
            time.asctime()))

    def train(self):
        xmp.spawn(self._mp_fn, args=(), nprocs=8)

    def _mp_fn(self, rank):
        torch.set_default_tensor_type('torch.FloatTensor')
        accuracy = self.main_func(rank)

    def main_func(self, rank):
        # 0. Data loader
        train_sampler = None
        test_sampler = None
        if xm.xrt_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=False)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=train_sampler,
                                      shuffle=False if train_sampler else True,
                                      batch_size=self.args.batch_size)
        test_dataloader = DataLoader(self.test_dataset,
                                     sampler=test_sampler,
                                     shuffle=False,
                                     batch_size=self.args.batch_size)

        # 1. Set device
        device = xm.xla_device()

        # 2. Set model
        model = self.model_class(self.bert_config, self.args).to(device)
        model.zero_grad()

        # 3. Setting the optimizer w/ new learning rate
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate * xm.xrt_world_size(), eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # 4. Set seed (minor)
        set_seed(self.args)

        def train_loop_fn(loader):
            if rank == 0:
                logger.info("***** Running training *****")
                logger.info("  Num examples = %d", len(self.train_dataset))
                logger.info("  Num Epochs = %d", self.args.num_train_epochs)
                logger.info("  Total train batch size = %d", self.args.batch_size)
                logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)

            tracker = xm.RateTracker()
            global_step = 0

            for step, batch in enumerate(loader):
                model.train()
                batch = tuple(t.to(device) for t in batch[1])  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    xm.optimizer_step(optimizer)
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        # Print the train status
                        xm.add_step_closure(self._train_update, args=(device, step, loss, tracker))

                    if rank == 0 and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    break

        def test_loop_fn(loader):
            if rank == 0:
                logger.info("***** Running evaluation on test dataset *****")
                logger.info("  Num examples = %d", len(self.test_dataset))
                logger.info("  Batch size = %d", self.args.batch_size)

            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None

            model.eval()

            for batch in loader:
                batch = tuple(t.to(device) for t in batch[1])
                with torch.no_grad():
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[3]}
                    if self.args.model_type != 'distilkobert':
                        inputs['token_type_ids'] = batch[2]
                    outputs = self.model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            results = {
                "loss": eval_loss
            }

            preds = np.argmax(preds, axis=1)
            result = compute_metrics(preds, out_label_ids)
            results.update(result)

            logger.info('[{}] Loss={:.2f} Accuracy={:.2f}%'.format(_get_device_spec(device), results['loss'], results['acc']))
            return results['accs']

        # 5. main function (train & test)
        accuracy = 0.0
        max_accuracy = 0.0

        for epoch in range(1, int(self.args.num_train_epochs) + 1):
            # Train
            para_loader = pl.ParallelLoader(train_dataloader, [device])
            train_loop_fn(para_loader.per_device_loader(device))
            xm.master_print('Finished training epoch {}'.format(epoch))

            # Test
            para_loader = pl.ParallelLoader(test_dataloader, [device])
            accuracy = test_loop_fn(para_loader.per_device_loader(device))
            max_accuracy = max(accuracy, max_accuracy)
            xm.master_print('Finished test epoch {}'.format(epoch))

        xm.master_print('Max Accuracy: {:.2f}%'.format(accuracy))
        return max_accuracy

    def save_model(self, model):
        # Save model checkpoint (Overwrite)
        output_dir = os.path.join(self.args.model_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert_config = self.config_class.from_pretrained(self.args.model_dir)
            logger.info("***** Config loaded *****")
            self.model = self.model_class.from_pretrained(self.args.model_dir, config=self.bert_config, args=self.args)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
