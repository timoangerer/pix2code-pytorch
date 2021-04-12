import platform
import os
import subprocess
import time
import torch
from pathlib import Path

# Taken from: https://github.com/yunjey/pytorch-tutorial/blob/0500d3df5a2a8080ccfccbc00aca0eacc21818db/tutorials/03-advanced/image_captioning/data_loader.py#L56


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def save_model(models_folder_path, encoder, decoder, optimizer, epoch, loss, batch_size, vocab):
    MODELS_FOLDER = Path(models_folder_path)

    # Create the models folder if it's not already there
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

    MODEL_PATH = MODELS_FOLDER / (model_name_formated("e-d-model",
                                  {"epoch": epoch, "loss": loss, "batch": batch_size}) + ".pth")

    torch.save({'epoch': epoch,
                'encoder_model_state_dict': encoder.state_dict(),
                'decoder_model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'vocab': vocab
                }, MODEL_PATH)

# Util for better model names when saving


def model_name_formated(model_name, stats_dict, delimiter="--"):
    current_time = time.strftime("%d-%m-%H-%M")
    stats_dict["time"] = current_time

    file_name = model_name

    for key, value in stats_dict.items():
        if isinstance(value, float):
            value = f'{value:.4f}'

        file_name = file_name + delimiter + str(key) + "-" + str(value)

    return file_name


def ids_to_tokens(vocab, ids):
    tokens = []

    for id in ids:
        token = vocab.get_token_by_id(id)

        if token == vocab.get_end_token():
            break
        if token == vocab.get_start_token() or token == ',':
            continue

        tokens.append(token)

    return tokens
