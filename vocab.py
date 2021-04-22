import json

START_TOKEN = "START"
END_TOKEN = "END"
PADDING_TOKEN = "PADDING"
UNKOWN_TOKEN = "UNKNOWN"


class Vocab():

    def __init__(self, vocab_path=None):

        init_tokens = [START_TOKEN, END_TOKEN, PADDING_TOKEN, UNKOWN_TOKEN]

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
            return self.id_to_token[self.token_to_id[UNKOWN_TOKEN]]
        else:
            return self.id_to_token[id]

    def get_id_by_token(self, token):
        if not token in self.token_to_id:
            return self.token_to_id[UNKOWN_TOKEN]
        else:
            return self.token_to_id[token]

    def get_start_token(self):
        return START_TOKEN

    def get_end_token(self):
        return END_TOKEN

    def get_padding_token(self):
        return PADDING_TOKEN

    def __str__(self):
        return json.dumps(self.token_to_id)

    def __len__(self):
        return len(self.token_to_id)
