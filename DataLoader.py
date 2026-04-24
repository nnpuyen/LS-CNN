import re
import random
from torch.utils.data import Dataset


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class MyData(Dataset):
    def __init__(self, cover_path=None, stego_path=None, examples=None):
        self.examples = []

        if examples is not None:
            self.examples = examples
        else:
            # load cover
            with open(cover_path, 'r', errors='ignore') as f:
                for line in f:
                    text = clean_str(line)
                    self.examples.append((text, 0))  # negative = 0

            # load stego
            with open(stego_path, 'r', errors='ignore') as f:
                for line in f:
                    text = clean_str(line)
                    self.examples.append((text, 1))  # positive = 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def sort_key(example):
        return len(example[0])

    @classmethod
    def split(cls, args, state, shuffle=True):
        if state == 'train':
            print('loading the training data...')
            dataset = cls(
                cover_path=args.train_cover_dir,
                stego_path=args.train_stego_dir
            )
            examples = dataset.examples

            if shuffle:
                random.shuffle(examples)

            # Lấy tỉ lệ validation từ args, mặc định 0.1
            val_ratio = getattr(args, 'valid_ratio', 0.1)
            n_total = len(examples)
            n_val = int(n_total * val_ratio)
            if n_val < 1:
                n_val = 1
            train_data = cls(examples=examples[:-n_val])
            val_data = cls(examples=examples[-n_val:])

            print(f"Tổng mẫu: {n_total}, train: {len(train_data)}, valid: {len(val_data)}, tỉ lệ valid: {val_ratio}")
            return train_data, val_data

        elif state == 'test':
            print('loading the testing data...')
            return cls(
                cover_path=args.test_cover_dir,
                stego_path=args.test_stego_dir
            )