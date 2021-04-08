import json


class Vocab():

    def __init__(self, vocab_path=None):
        init_tokens = ["<s>", "</s>", "<pad>", "<unk>"]

        self.token_to_id = dict()
        self.id_to_token = dict()

        for token in init_tokens:
            self.add_token(token)

        if vocab_path:
            tokens = self.__read_vocab_from_file__(vocab_path)
            for token in tokens:
                self.add_token(token)

    def add_token(self, token):
        if token not in self.token_to_id:
            curr_length = len(self.token_to_id)
            self.token_to_id[token] = curr_length
            self.id_to_token[curr_length] = token

    def __read_vocab_from_file__(self, vocab_path):
        with open(vocab_path, "r") as reader:
            tokens = reader.read().split(' ')
            return tokens

    def get_token_by_id(self, id):
        if not id in self.id_to_token:
            return self.id_to_token["<unk>"]
        else:
            return self.id_to_token[id]

    def get_id_by_token(self, token):
        if not token in self.token_to_id:
            return self.token_to_id["<unk>"]
        else:
            return self.token_to_id[token]

    def __str__(self):
        return json.dumps(self.token_to_id)

    def __len__(self):
        return len(self.token_to_id)
