import os
import glob
import subprocess

# Định nghĩa các domain và đường dẫn

# base_dir: thư mục cha của LS-CNN
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
domains = [
    {
        'name': 'IMDB',
        'folder': os.path.join(base_dir, 'IMDB', 'IMDB-95k'),
    },
    {
        'name': 'Twitter',
        'folder': os.path.join(base_dir, 'Twitter'),
    },
    {
        'name': 'News',
        'folder': os.path.join(base_dir, 'News', 'News-95k'),
    },
    {
        'name': 'THU-Stega',
        'folder': os.path.join(base_dir, 'THU-Stega'),
    },
]

snapshot_dir = os.path.join(base_dir, 'LS-CNN', 'snapshot')
build_vocab_py = os.path.join(base_dir, 'build_vocab.py')
main_py = os.path.join(os.path.dirname(__file__), 'main.py')

# 1. Build vocab cho từng domain
for domain in domains:
    data_files = [
        os.path.join(domain['folder'], 'cover_train.txt'),
        os.path.join(domain['folder'], 'stego_train.txt'),
    ]
    vocab_path = os.path.join(snapshot_dir, f"vocab_{domain['name']}.pkl")
    cmd = [
        'python', build_vocab_py,
        '--data_files', *data_files,
        '--out_vocab', vocab_path
    ]
    print(f"\n==== Building vocab for {domain['name']} ====")
    subprocess.run(cmd, check=True)

# 2. Train model trên từng domain
for domain in domains:
    train_cover = os.path.join(domain['folder'], 'cover_train.txt')
    train_stego = os.path.join(domain['folder'], 'stego_train.txt')
    test_cover = os.path.join(domain['folder'], 'cover_test.txt')
    test_stego = os.path.join(domain['folder'], 'stego_test.txt')
    save_dir = os.path.join(snapshot_dir, domain['name'])
    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        'python', main_py,
        '-train-cover-dir', train_cover,
        '-train-stego-dir', train_stego,
        '-test-cover-dir', test_cover,
        '-test-stego-dir', test_stego,
        '-save-dir', save_dir
    ]
    print(f"\n==== Training and in-domain evaluation for {domain['name']} ====")
    subprocess.run(cmd, check=True)

# 3. Cross-domain evaluation
for src in domains:
    src_save_dir = os.path.join(snapshot_dir, src['name'])
    # Tìm checkpoint tốt nhất (ưu tiên best_steps_*.pt)
    ckpt_list = glob.glob(os.path.join(src_save_dir, 'best_steps_*.pt'))
    if not ckpt_list:
        print(f"No checkpoint found for {src['name']}, skipping cross-domain eval.")
        continue
    ckpt_path = sorted(ckpt_list)[-1]
    for tgt in domains:
        if src['name'] == tgt['name']:
            continue
        test_cover = os.path.join(tgt['folder'], 'cover_test.txt')
        test_stego = os.path.join(tgt['folder'], 'stego_test.txt')
        cmd = [
            'python', main_py,
            '-test', 'True',
            '-load_dir', ckpt_path,
            '-test-cover-dir', test_cover,
            '-test-stego-dir', test_stego,
            '-save-dir', src_save_dir
        ]
        print(f"\n==== Cross-domain: {src['name']} model -> {tgt['name']} test ====")
        subprocess.run(cmd, check=True)

print("\nPipeline hoàn tất!")
