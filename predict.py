import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification

from utils import init_logger, load_tokenizer

logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)  # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def convert_input_file_to_tensor_dataset(pred_config,
                                         args,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    tokenizer = load_tokenizer(args)

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []

    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            tokens = tokenizer.tokenize(line)
            # Account for [CLS] and [SEP]
            special_tokens_count = 2
            if len(tokens) > args.max_seq_len - special_tokens_count:
                tokens = tokens[:(args.max_seq_len - special_tokens_count)]

            # Add [SEP] token
            tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = args.max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

    return dataset


def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    logger.info(args)

    # Convert input file to TensorDataset
    dataset = convert_input_file_to_tensor_dataset(pred_config, args)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)

    # Write to output file
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for pred in preds:
            f.write("{}\n".format(pred))

    logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="sample_pred_in.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    predict(pred_config)
