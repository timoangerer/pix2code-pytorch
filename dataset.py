from pathlib import Path
from PIL import Image
import torch

class Pix2CodeDataset():

    def __init__(self, data_path, split, vocab, transform=None):
        assert split in ["train", "validation", "test"]
        self.data_path = data_path
        self.transform = transform
        self.vocab = vocab

        self.filenames = []
        with open(Path(Path(data_path).parent, f'{split}_dataset.txt'), "r") as reader:
            self.filenames = reader.read().split('\n')
            self.filenames.remove('')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = Path(self.data_path, self.filenames[idx] + ".png")
        tokens_path = Path(self.data_path, self.filenames[idx] + ".gui")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        tokens = self.parse_gui_token_file(tokens_path)
        tokens.insert(0, self.vocab.get_start_token())
        tokens.append(self.vocab.get_end_token())

        token_ids = [self.vocab.get_id_by_token(token) for token in tokens]

        token_ids = torch.Tensor(token_ids)

        return image, token_ids


    def parse_gui_token_file(self, filepath):
        suffix = filepath.suffix

        assert suffix == ".gui"

        with open(filepath, "r") as reader:
            raw_data = reader.read()
            tokens = raw_data.replace('\n', ' ').replace(
                ', ', ' , ').split(' ')
            tokens.remove('')

        return tokens
