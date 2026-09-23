"""SHU-MI adaptation of the authors' finetune_main.py.

Keeps the original CLI arguments and SHU-MI execution branch. Uses the already
installed original trainer/evaluator/model with CPU portability changes.
Other dataset branches are omitted because their data loaders are not installed.
Source: wjq-learning/CBraMod, revision b9e961003214326972c567eff390e75b0287e32a.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from preprocessing import shu_dataset
from finetune_trainer import Trainer
from models.CBraMod import model_for_shu

PROJECT = Path(__file__).resolve().parent


def parse_bool(value):
    # argparse's type=bool treats the string "False" as True. Parse explicitly.
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Use True or False.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--model', choices=['cbramod', 'simpleconv'], default='cbramod')
    parser.add_argument('--simpleconv_fm', type=int, default=128)
    parser.add_argument('--simpleconv_n_convs', type=int, default=4)
    parser.add_argument('--simpleconv_kernel_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='SHU-MI',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, '
                             'SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a]')
    parser.add_argument('--datasets_dir', type=str,
                        default=str(PROJECT / 'data/preprocessed'),
                        help='datasets_dir')
    parser.add_argument('--train_fraction', type=float, default=1.0, help='Fraction of training trials, in (0, 1].')
    parser.add_argument('--train_keys', default=None, help='Optional JSON training-key manifest.')
    parser.add_argument('--num_of_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--model_dir', type=str, default=str(PROJECT / 'runs/cbramod_main_seed3407'), help='model_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=parse_bool, default=True,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--frozen', type=parse_bool,
                        default=False, help='frozen')
    parser.add_argument('--use_pretrained_weights', type=parse_bool,
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default=str(PROJECT / 'models/model_weights/CBraMod/pretrained_weights.pth'),
                        help='foundation_dir')

    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--check-only', action='store_true',
                        help='Load weights and check a real batch without training or saving.')
    params = parser.parse_args(argv)
    if not 0 < params.train_fraction <= 1:
        parser.error('train_fraction must be in (0, 1].')
    if min(params.simpleconv_fm, params.simpleconv_n_convs, params.simpleconv_kernel_size) < 1:
        parser.error('SimpleConv architecture parameters must be positive.')
    if params.model == 'simpleconv':
        if params.multi_lr or params.frozen:
            parser.error('EEGSimpleConv comparison requires --multi_lr False --frozen False.')
        # EEGSimpleConv is a supervised baseline initialized from scratch.
        params.use_pretrained_weights = False
        params.foundation_dir = None
    if params.downstream_dataset != 'SHU-MI':
        parser.error('This project adaptation supports SHU-MI only.')
    if params.epochs < 1 or params.batch_size < 1:
        parser.error('epochs and batch_size must be positive.')
    if params.classifier not in ('all_patch_reps', 'all_patch_reps_twolayer',
                                 'all_patch_reps_onelayer', 'avgpooling_patch_reps'):
        parser.error('Unknown classifier.')
    if params.device == 'auto':
        params.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if params.device == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable. Use --device cpu.')
        torch.cuda.set_device(params.cuda)
        params.device = f'cuda:{params.cuda}'
    if not params.check_only and Path(params.model_dir).exists():
        parser.error('model_dir already exists; choose a new folder to preserve old results.')
    if params.use_pretrained_weights:
        # The original Model loads this local Hugging Face backbone checkpoint.
        params.foundation_dir = str(Path(params.foundation_dir).expanduser().resolve())
        if not Path(params.foundation_dir).is_file():
            parser.error('Pretrained weights not found. Download them first or set --foundation_dir.')

    setup_seed(params.seed)
    torch.set_num_threads(4)
    print(params)

    # This is the original SHU-MI branch of finetune_main.py.
    load_dataset = shu_dataset.LoadDataset(params)
    data_loader = load_dataset.get_data_loader()
    if params.model == 'cbramod':
        model = model_for_shu.Model(params)
    else:
        from models.EEGSimpleConv.model_for_shu_simpleconv import Model
        model = Model(params)

    if params.check_only:
        model = model.to(params.device).eval()
        x, _ = next(iter(data_loader['train']))
        with torch.no_grad():
            logits = model(x[:2].to(params.device))
        assert logits.shape == (min(2, len(x)),)
        print('Model and real-batch forward check passed:', tuple(logits.shape))
        return

    # New output folder per seed/experiment. Validation AUROC selects the model.
    Path(params.model_dir).mkdir(parents=True, exist_ok=False)
    (Path(params.model_dir) / 'config.json').write_text(json.dumps(vars(params), indent=2))
    (Path(params.model_dir) / 'training_keys.json').write_text(
        json.dumps(data_loader['train'].dataset.keys))
    t = Trainer(params, data_loader, model)
    result = t.train_for_binaryclass()
    result.update(model=params.model, seed=params.seed, epochs=params.epochs,
                  scope='full' if params.epochs == 50 else 'pilot',
                  multi_lr=params.multi_lr, device=params.device,
                  pretrained=params.use_pretrained_weights, frozen=params.frozen,
                  python=__import__('sys').version.split()[0], torch=torch.__version__)
    result['train_keys'] = params.train_keys
    result['train_fraction'] = params.train_fraction
    result['split_sizes'] = {k: len(v.dataset) for k, v in data_loader.items()}
    result['training_protocol'] = {
        name: getattr(params, name) for name in
        ['epochs', 'batch_size', 'lr', 'weight_decay', 'optimizer', 'clip_value', 'multi_lr']
    }
    if params.model == 'simpleconv':
        result['architecture'] = dict(fm=params.simpleconv_fm, n_convs=params.simpleconv_n_convs, kernel_size=params.simpleconv_kernel_size,
                                      sfreq=200, resampling=200, n_chan=32, n_classes=1)
    (Path(params.model_dir) / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    main()
