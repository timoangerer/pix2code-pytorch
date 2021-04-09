import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Split the dataset into train, validation and test datasets')

parser.add_argument("--data_path", type=str,
                        default=Path("data", "screenshot-description-pairs"), help="Datapath")
parser.add_argument("--vocab_output_path", type=str,
                        default=Path("data", "vocab.txt"), help="Output path of the vocab file")            

args = parser.parse_args()
data_path = args.data_path
vocab_output_path = args.vocab_output_path

all_tokens = set()
for file in Path(data_path).iterdir():
    stem = file.stem
    suffix = file.suffix

    if suffix == ".gui":
        with open(file, "r") as reader:
            raw_data = reader.read()
            data = raw_data.replace('\n', ' ').replace(', ', ' , ').split(' ')
            data.remove('')
            [all_tokens.add(el) for el in data]  


# write the set of all tokens to a vocab file
print(f'Writing vocab with {len(all_tokens)} tokens')

with open(vocab_output_path, "w") as writer:
    writer.write(" ".join(all_tokens))