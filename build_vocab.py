import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Generate the vocabulary file based on the specifed dataset')

parser.add_argument("--data_path", type=str,
                        default=Path("data", "web", "all_data"), help="Datapath")
parser.add_argument("--vocab_output_path", type=str,
                        default=None, help="Output path of the vocab file")            

args = parser.parse_args()
args.vocab_output_path = args.vocab_output_path if args.vocab_output_path else Path(Path(args.data_path).parent, "vocab.txt")

data_path = Path(args.data_path)
vocab_output_path = args.vocab_output_path

all_tokens = dict() # dict used as ordered set since it preserves order
for file in Path(data_path).iterdir():
    stem = file.stem
    suffix = file.suffix

    if suffix == ".gui":
        with open(file, "r") as reader:
            raw_data = reader.read()
            data = raw_data.replace('\n', ' ').replace(', ', ' , ').split(' ')
            data.remove('')
            for el in data:
                all_tokens[el] = el 


# write the set of all tokens to a vocab file
print(f'Writing vocab with {len(all_tokens)} tokens')

with open(vocab_output_path, "w") as writer:
    writer.write(" ".join(all_tokens))