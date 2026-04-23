import os
import sys
import csv
from datetime import datetime
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def _append_csv_row(file_path, headers, row):
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _append_txt_line(file_path, text):
    with open(file_path, 'a', encoding='utf-8', errors='ignore') as f:
        f.write(text + '\n')


def _log_training_step(args, epoch, step, loss, acc):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    csv_path = os.path.join(args.save_dir, 'training_process.csv')
    txt_path = os.path.join(args.save_dir, 'training_process.txt')
    headers = ['timestamp', 'epoch', 'step', 'loss', 'accuracy']

    _append_csv_row(
        csv_path,
        headers,
        {
            'timestamp': now,
            'epoch': epoch,
            'step': step,
            'loss': f'{loss:.6f}',
            'accuracy': f'{acc:.6f}',
        },
    )
    _append_txt_line(
        txt_path,
        f'[{now}] epoch={epoch} step={step} loss={loss:.6f} accuracy={acc:.6f}',
    )


def _log_evaluation(args, phase, loss, acc, precision='', recall='', f1='', tn='', tp='', fn='', fp='', dataset='', P_FA='', P_MD='', P_E=''):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    csv_path = os.path.join(args.save_dir, 'evaluation_results.csv')
    txt_path = os.path.join(args.save_dir, 'evaluation_results.txt')
    headers = [
        'timestamp', 'phase', 'dataset', 'loss', 'accuracy', 'precision', 'recall', 'f1',
        'tn', 'tp', 'fn', 'fp', 'P_FA', 'P_MD', 'P_E'
    ]

    row = {
        'timestamp': now,
        'phase': phase,
        'dataset': dataset,
        'loss': f'{loss:.6f}',
        'accuracy': f'{acc:.6f}',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tn': tn,
        'tp': tp,
        'fn': fn,
        'fp': fp,
        'P_FA': P_FA,
        'P_MD': P_MD,
        'P_E': P_E,
    }
    _append_csv_row(csv_path, headers, row)
    _append_txt_line(
        txt_path,
        f'[{now}] phase={phase} dataset={dataset} loss={loss:.6f} accuracy={acc:.6f} precision={precision} recall={recall} f1={f1} tn={tn} tp={tp} fn={fn} fp={fp} P_FA={P_FA} P_MD={P_MD} P_E={P_E}',
    )

def train(train_iter, dev_iter, model, args, device):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
        weight_decay=1e-6
    )

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'LogFile'))
    os.makedirs(args.save_dir, exist_ok=True)

    steps = 0
    best_acc = 0
    last_step = 0

    model.train()

    for epoch in range(1, args.epochs + 1):
        print(f'\n--------training epochs: {epoch}-----------')

        for batch_idx, batch in enumerate(train_iter, 1):
            feature, target = batch
            feature = feature.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()

            logit = model(feature)
            loss = F.cross_entropy(logit, target)

            loss.backward()
            optimizer.step()

            steps += 1

            if steps % args.log_interval == 0:
                pred = logit.argmax(dim=1)
                corrects = (pred == target).sum().item()
                accuracy = corrects / target.size(0)

                print(f'\rBatch[{batch_idx}] - loss:{loss.item():.6f} acc:{accuracy:.4f}', end='')

                writer.add_scalar('loss/train', loss.item(), steps)
                writer.add_scalar('acc/train', accuracy, steps)
                _log_training_step(args, epoch, steps, loss.item(), accuracy)

            if steps % args.test_interval == 0:
                dev_acc, dev_loss, dev_precision, dev_recall, dev_f1, dev_P_FA, dev_P_MD, dev_P_E = data_eval(dev_iter, model, args, device)

                writer.add_scalar('loss/dev', dev_loss, steps)
                writer.add_scalar('acc/dev', dev_acc, steps)
                writer.add_scalar('precision/dev', dev_precision, steps)
                writer.add_scalar('recall/dev', dev_recall, steps)
                writer.add_scalar('f1/dev', dev_f1, steps)
                writer.add_scalar('P_FA/dev', dev_P_FA, steps)
                writer.add_scalar('P_MD/dev', dev_P_MD, steps)
                writer.add_scalar('P_E/dev', dev_P_E, steps)

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)

                if steps - last_step >= args.early_stop:
                    print('early stop')
                    return

                model.train()

def data_eval(data_iter, model, args, device):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():  # 🔥 QUAN TRỌNG
        for batch in data_iter:
            feature, target = batch
            feature = feature.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logit = model(feature)
            loss = F.cross_entropy(logit, target)

            # loss
            total_loss += loss.item()

            # accuracy
            pred = logit.argmax(dim=1)
            total_correct += (pred == target).sum().item()
            total_samples += target.size(0)

            all_preds.append(pred.cpu())
            all_labels.append(target.cpu())

    avg_loss = total_loss / len(data_iter)
    accuracy = total_correct / total_samples

    import numpy as np
    from sklearn import metrics
    predictions = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    acc = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = metrics.recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = metrics.f1_score(labels, predictions, average='weighted')

    TN = ((predictions == 0) & (labels == 0)).sum()
    TP = ((predictions == 1) & (labels == 1)).sum()
    FN = ((predictions == 0) & (labels == 1)).sum()
    FP = ((predictions == 1) & (labels == 0)).sum()

    # P_FA: False Alarm Rate = FP / (FP + TN)
    P_FA = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    # P_MD: Missed Detection = FN / (FN + TP)
    P_MD = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    # P_E: Detection Error Rate = (P_FA + P_MD) / 2
    P_E = (P_FA + P_MD) / 2

    phase = 'test' if args.test else 'validation'
    print(f"\n{phase.capitalize()} - loss:{avg_loss:.6f} acc:{acc:.4f} ({total_correct}/{total_samples})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"P_FA (False Alarm Rate): {P_FA:.4f}")
    print(f"P_MD (Missed Detection): {P_MD:.4f}")
    print(f"P_E (Detection Error Rate): {P_E:.4f}")

    dataset_name = getattr(args, 'test_dataset_name', '') if args.test else getattr(args, 'train_dataset_name', '')
    _log_evaluation(
        args,
        phase=phase,
        loss=avg_loss,
        acc=acc,
        precision=f'{precision:.6f}',
        recall=f'{recall:.6f}',
        f1=f'{f1:.6f}',
        tn=int(TN),
        tp=int(TP),
        fn=int(FN),
        fp=int(FP),
        dataset=dataset_name,
        P_FA=f'{P_FA:.6f}',
        P_MD=f'{P_MD:.6f}',
        P_E=f'{P_E:.6f}'
    )

    # Lưu ra file txt
    result_file = os.path.join(args.save_dir, f'{phase}_result.txt')
    with open(result_file, 'a', errors='ignore') as f:
        f.write(f'{phase.capitalize()} accuracy: {acc:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1: {f1:.4f}\n')
        f.write(f'P_FA: {P_FA:.4f}\n')
        f.write(f'P_MD: {P_MD:.4f}\n')
        f.write(f'P_E: {P_E:.4f}\n')

    return acc, avg_loss, precision, recall, f1, P_FA, P_MD, P_E

def save(model, save_dir, save_prefix, steps):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	save_prefix = os.path.join(save_dir, save_prefix)
	save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
	torch.save(model.state_dict(), save_path)

