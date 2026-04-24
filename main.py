import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
import requests
import zipfile
from torch.utils.data import DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random

import LS_CNN
import train
import DataLoader as DL
from DataLoader import MyData


# Đặt random seed để đảm bảo reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# tokenizer
def tokenize(text):
    return text.split()

# build vocab
def build_vocab(dataset, min_freq=1):
    counter = Counter()
    for text, _ in dataset.examples:
        counter.update(tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab

# encode
def encode(text, vocab):
    return torch.tensor(
        [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)],
        dtype=torch.long
    )

# load GloVe
def load_glove(vocab, glove_path, dim=300):
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), dim))

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=float)

            if word in vocab:
                embeddings[vocab[word]] = vector

    return torch.tensor(embeddings, dtype=torch.float)

# collate_fn
def collate_fn(batch, vocab):
    texts, labels = zip(*batch)

    encoded_texts = [encode(text, vocab) for text in texts]
    padded_texts = pad_sequence(
        encoded_texts,
        batch_first=True,
        padding_value=vocab["<pad>"]
    )

    labels = torch.tensor(labels, dtype=torch.long)

    return padded_texts, labels

class CollateWithVocab:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        batch = sorted(batch, key=lambda x: len(x[0].split()), reverse=True)
        return collate_fn(batch, self.vocab)

def data_loader(args, batch_size=32, shuffle=True, glove_path=r"glove_weight/glove.6B.300d.txt"):
    train_data, valid_data = MyData.split(args, state='train')
    print(f"Batch size in args: {args.batch_size}")
    print(f"Batch size in function: {batch_size}")
    vocab = build_vocab(train_data)
    
    if os.path.exists("glove_weight/glove.6B.300d.txt.pt"):
        embedding_matrix = torch.load("glove_weight/glove.6B.300d.txt.pt")
    else:
        embedding_matrix = torch.tensor(load_glove(vocab, glove_path),dtype=torch.float)
        torch.save(embedding_matrix, "glove_weight/glove.6B.300d.txt.pt")
        
    collate = CollateWithVocab(vocab)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=len(valid_data),
        shuffle=False,
        collate_fn=collate
    )
    
    return train_loader, valid_loader, vocab, embedding_matrix

def infer_dataset_name(cover_path, stego_path):
    def normalize_name(path):
        normalized = os.path.normpath(path)
        if os.path.isdir(normalized):
            return os.path.basename(normalized)
        return os.path.splitext(os.path.basename(normalized))[0]

    cover_name = normalize_name(cover_path)
    stego_name = normalize_name(stego_path)

    if cover_name == stego_name:
        return cover_name
    return f"{cover_name} + {stego_name}"

def print_result_table(title, metrics_dict):
    print(f"\n{title}")
    print("+-----------+----------+")
    print("| Metric    | Value    |")
    print("+-----------+----------+")
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            formatted_value = f"{value:>8.4f}"
        else:
            formatted_value = f"{str(value):>8}"
        print(f"| {key:<9} | {formatted_value} |")
    print("+-----------+----------+")

def eval_and_report(model, test_loader, args, device, title, dataset_name):
    previous_test_flag = args.test
    args.test = True
    acc, _, precision, recall, f1, P_FA, P_MD, P_E = train.data_eval(test_loader, model, args, device)
    args.test = previous_test_flag

    print(f"Dataset: {dataset_name}")
    metrics_dict = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "P_FA": P_FA,
        "P_MD": P_MD,
        "P_E": P_E,
    }
    print_result_table(title, metrics_dict)


if __name__ == "__main__":
    set_seed(42)

    # Đường dẫn lưu file
    folder = "glove_weight"
    os.makedirs(folder, exist_ok=True)
    glove_file_path = os.path.join(folder, "glove.6B.300d.txt")

    if not os.path.exists(glove_file_path):
        # URL của file zip chứa glove.6B.300d.txt
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = os.path.join(folder, "glove.6B.zip")

        # Tải file zip
        print("Downloading GloVe embeddings...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download completed.")

        # Giải nén glove.6B.300d.txt
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract("glove.6B.300d.txt", folder)
        print("Extracted glove.6B.300d.txt.")

        # Xóa file zip nếu muốn
        os.remove(zip_path)
        print("Cleanup done.")
    else:
        print("GloVe embeddings already exist. Skipping download.")

    parser = argparse.ArgumentParser(description='LS_CNN')

    # learning
    parser.add_argument('-batch-size', type=int, default=64, \
                        help='batch size for training [default: 64]')
    parser.add_argument('-lr', type=float, default=0.001,\
                        help='initial learning rate [default:0.001]')
    parser.add_argument('-epochs', type=int, default=20,\
                        help='number of epochs for train [default:20]')
    parser.add_argument('-log-interval', type=int, default=20, \
                        help='how many steps to wait defore logging train status')
    parser.add_argument('-test-interval', type=int, default=100, \
                        help='how many steps to wait defore testing [default:100]')
    parser.add_argument('-save-interval', type=int, default=500, \
                        help='how many steps to wait before saving [default:500]')
    parser.add_argument('-early-stop', type=int, default=1000, \
                        help='iteration numbers to stop without performace boost')
    parser.add_argument('-save-best', type=bool, default=True,\
                        help='whether to save when get best performance')
    parser.add_argument('-save-dir', type=str, default='snapshot',
                        help='where to save the snapshot')
    parser.add_argument('-load_dir', type=str, default=None,
                        help='where to loading the trained model')

    # data
    parser.add_argument('-shuffle', action='store_true', default=True,\
                        help='shuffle the data every epoch [default:True]')
    parser.add_argument('-train-cover-dir', type=str, default=r'..\\dataset-10k\\filtered\\cover_train.txt',
                        help='the path of train cover data. [default:cover.txt]')
    parser.add_argument('-train-stego-dir', type=str, default=r'..\\dataset-10k\\filtered\\stego_train.txt',
                        help='the path of train stego data. [default:1bpw.txt]')
    parser.add_argument('-test-cover-dir', type=str, default=r'..\\dataset-10k\\filtered\\cover_test.txt',
                        help='the path of test cover data. [default:cover.txt]')
    parser.add_argument('-test-stego-dir', type=str, default=r'..\\dataset-10k\\filtered\\stego_test.txt',
                        help='the path of test stego data. [default:1bpw.txt]')
    # model
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5', \
                        help='vomma-speparated kernel size to use for convolution')
    parser.add_argument('-embed-dim', type=int, default=300, \
                        help='number of embedding dimension [defualt:300]')
    parser.add_argument('-kernel-num', type=int, default=100, \
                        help='number of each kind of kernel [defualt:100]')
    parser.add_argument('-dropout', type=float, default=0.5, \
                        help='the probability for dropout [defualt:0.5]')
    parser.add_argument('-static', action='store_true', default=False, \
                        help='fix the embedding [default:False]')

    #device
    parser.add_argument('-no-cuda', action='store_true', default=False, \
                        help='disable the gpu [default:True]')
    parser.add_argument('-device', type=str, default='cuda', \
                        help='device to use for trianing [default:gpu]')
    parser.add_argument('-idx-gpu', type=str, default='0',\
                        help='the number of gpu for training [default:0]')

    # option
    parser.add_argument('-test', type=bool, default=False, \
                        help='train or test [default:False]')                              

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.idx_gpu

    # Load data
    train_loader, valid_loader, vocab, embedding_matrix = data_loader(args)
    print(f"Số batch trong train_iter: {len(train_loader)}")
    print(f"Số batch trong dev_iter: {len(valid_loader)}")

    # Test data (always create so we can evaluate right after training)
    test_dataset_name = infer_dataset_name(args.test_cover_dir, args.test_stego_dir)
    args.test_dataset_name = test_dataset_name
    test_data = MyData.split(args, state='test')
    
    collate = CollateWithVocab(vocab)
    test_loader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False,
        collate_fn=collate
    )

    # Update args
    args.embed_num = len(vocab)
    args.class_num = 2
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.kernel_sizes = list(map(int, args.kernel_sizes.split(',')))

    # 🔹 Model
    model = LS_CNN.LS_CNN(args, embedding_matrix)

    # 🔹 Load pretrained embedding
    print(type(embedding_matrix))
    print(len(embedding_matrix))
    print(type(embedding_matrix[0]))
    print(embedding_matrix[0])
    print(embedding_matrix.shape if hasattr(embedding_matrix, 'shape') else 'no shape')
        
    # 🔹 Initialize model
    for name, w in model.named_parameters():
        if 'embed_' not in name:
            if 'fc1.weight' in name:
                nn.init.xavier_normal_(w)
            elif 'weight' in name:
                nn.init.normal_(w, 0.0, 0.1)
            elif 'bias' in name:
                nn.init.constant_(w, 0)

    # 🔹 Load checkpoint nếu có
    if args.load_dir is not None:
        print(f'\nLoading model from {args.load_dir}...')
        model.load_state_dict(torch.load(args.load_dir))

    # 🔹 Device
    device = torch.device(args.device if args.cuda else 'cpu')
    model = model.to(device)

    # 🔹 Training phase
    if not args.test:
        train.train(train_loader, valid_loader, model, args, device)
        eval_and_report(model, test_loader, args, device, 'Final model test result', test_dataset_name)

    # 🔹 Testing phase
    else:
        print('\n----------testing------------')
        # Nếu truyền --load_dir thì chỉ test đúng checkpoint đó
        if args.load_dir is not None:
            print(f'Loading test model from {args.load_dir}...')
            if not os.path.exists(args.load_dir):
                print(f'Cannot find checkpoint: {args.load_dir}')
                sys.exit(1)
            model.load_state_dict(torch.load(args.load_dir, map_location=device))
            eval_and_report(model, test_loader, args, device, f'Test result for {os.path.basename(args.load_dir)}', test_dataset_name)
        else:
            print(f'Loading test model from {args.save_dir}...')
            if not os.path.exists(args.save_dir):
                print(f'Cannot find save directory: {args.save_dir}')
                sys.exit(1)

            models = []
            files = sorted(os.listdir(args.save_dir))
            for name in files:
                if name.endswith('.pt'):
                    models.append(name)

            if not models:
                print(f'No checkpoint found in {args.save_dir}')
                sys.exit(1)

            model_steps = sorted([
                int(m.split('_')[-1].split('.')[0]) for m in models
            ])

            for step in model_steps[-3:]:
                best_model = f'best_steps_{step}.pt'
                m_path = os.path.join(args.save_dir, best_model)

                print(f'the {m_path} model is loaded...')
                model.load_state_dict(torch.load(m_path, map_location=device))
                eval_and_report(model, test_loader, args, device, f'Test result for {best_model}', test_dataset_name)