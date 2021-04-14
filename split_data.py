import argparse
from pathlib import Path
from math import floor

parser = argparse.ArgumentParser(description='Split the dataset into train, validation and test datasets')

parser.add_argument("--data_path", type=str,
                        default=Path("data", "web", "all_data"), help="Datapath")

args = parser.parse_args()
data_path = Path(args.data_path)

TRAIN_PERCENT = 0.6
TEST_PERCENT = 0.2
VALIDATION_PERCENT = 0.2

occurences_count = dict()
for file in Path(data_path).iterdir():
    stem = file.stem
    suffix = file.suffix

    if stem not in occurences_count:
        count = {}
        count[suffix] = 1
        occurences_count[stem] = count
    else:
        occurences_count[stem][suffix] = occurences_count[stem][suffix] =+ 1

# map to array only containing valid pairs
valid_pairs = []
for key, value in occurences_count.items():
    try:
        if value[".gui"] == 1 and value[".png"] == 1:
            valid_pairs.append(key)
    except:
        print(f'File {key} is not a valid pair')



number_of_examples = len(valid_pairs)
print(f'Found a total of {number_of_examples} valid examples')

train_split = floor(number_of_examples * TRAIN_PERCENT)
validation_split = floor(number_of_examples * VALIDATION_PERCENT)
test_split = floor(number_of_examples * TEST_PERCENT)

train_set = valid_pairs[:train_split]
validation_set = valid_pairs[train_split: train_split + validation_split]
test_set = valid_pairs[train_split + validation_split:]

dataset_splits = {"train": train_set, "validation": validation_set, "test": test_set}

for key, value in dataset_splits.items():
    filepath = Path(data_path.parent, f'{key}_dataset.txt')

    with open(filepath, "w") as writer:
        for example in value:
            writer.write(example + "\n")