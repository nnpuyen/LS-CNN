import os
import random
import argparse

# Hàm lấy tất cả các file có tiền tố 'test' trong thư mục cover
def get_test_files(cover_dir):
    return [os.path.join(cover_dir, f) for f in os.listdir(cover_dir) if f.startswith('test') and os.path.isfile(os.path.join(cover_dir, f))]

def read_all_lines(files):
    lines = []
    for file in files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            lines.extend([line.strip() for line in f if line.strip()])
    return lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly select 4000 cover samples for each class from test cover files.")
    parser.add_argument('--cover_dir', type=str, required=True, help='Path to the cover directory')
    parser.add_argument('--output', type=str, default='test_cover_4000.txt', help='Output file for selected covers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    test_files = get_test_files(args.cover_dir)
    if not test_files:
        print(f"No test files found in {args.cover_dir}")
        exit(1)

    all_lines = read_all_lines(test_files)
    random.shuffle(all_lines)
    n = min(4000, len(all_lines))
    selected = all_lines[:n]

    with open(args.output, 'w', encoding='utf-8') as f:
        for line in selected:
            f.write(line + '\n')

    print(f"Saved {n} random samples to {args.output}")
