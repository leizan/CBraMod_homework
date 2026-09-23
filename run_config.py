"""Run the existing finetune_main.py with a readable JSON experiment configuration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

TRAINING = {'epochs', 'batch_size', 'lr', 'weight_decay', 'optimizer', 'clip_value', 'multi_lr', 'device'}
MODEL_PARAMETERS = {
    'cbramod': {'use_pretrained_weights', 'foundation_dir', 'classifier', 'dropout', 'frozen'},
    'simpleconv': {'simpleconv_fm', 'simpleconv_n_convs', 'simpleconv_kernel_size'},
}


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    required = {'model', 'datasets_dir', 'seeds', 'output_dir', 'training', 'model_parameters'}
    if not required <= set(cfg) or set(cfg) - required - {'train_fraction'}:
        raise ValueError(f'Config requires {sorted(required)}; train_fraction is optional.')
    fraction = cfg.get('train_fraction', 1.0)
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        raise ValueError('train_fraction must be a number in (0, 1].')
    if cfg['model'] not in MODEL_PARAMETERS:
        raise ValueError('Unknown model.')
    if set(cfg['training']) != TRAINING:
        raise ValueError(f'Training keys must be {sorted(TRAINING)}')
    if set(cfg['model_parameters']) != MODEL_PARAMETERS[cfg['model']]:
        raise ValueError(f'Wrong model_parameters for {cfg["model"]}')
    seeds = cfg['seeds']
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be a nonempty list of distinct nonnegative integers.')
    t = cfg['training']
    for key in ('epochs', 'batch_size'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if type(t['multi_lr']) is not bool:
        raise ValueError('multi_lr must be JSON true or false.')
    if t['lr'] <= 0 or t['weight_decay'] < 0 or t['clip_value'] < 0:
        raise ValueError('Invalid learning rate, weight decay, or clipping value.')
    if t['optimizer'] not in ('AdamW', 'SGD') or t['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Unsupported optimizer or device.')
    mp = cfg['model_parameters']
    if cfg['model'] == 'simpleconv':
        if t['multi_lr'] or any(type(v) is not int or v < 1 for v in mp.values()):
            raise ValueError('SimpleConv requires multi_lr=false and positive integer architecture parameters.')
    else:
        if any(type(mp[k]) is not bool for k in ('frozen', 'use_pretrained_weights')):
            raise ValueError('frozen and use_pretrained_weights must be JSON booleans.')
        if not 0 <= mp['dropout'] < 1:
            raise ValueError('dropout must be in [0,1).')
    return cfg


def resolve_path(project, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(project) / path


def make_command(project, cfg, seed, check_only=False):
    project = Path(project).resolve()
    args = {'model': cfg['model'], 'datasets_dir': str(resolve_path(project, cfg['datasets_dir'])),
            'seed': seed, 'model_dir': str(resolve_path(project, cfg['output_dir']) / f'seed_{seed}'),
            **cfg['training'], **cfg['model_parameters']}
    args['train_fraction'] = cfg.get('train_fraction', 1.0)
    if cfg['model'] == 'simpleconv':
        args['use_pretrained_weights'] = False
    elif args['use_pretrained_weights']:
        args['foundation_dir'] = str(resolve_path(project, args['foundation_dir']))
    command = [sys.executable, str(project / 'finetune_main.py')]
    for key, value in args.items():
        command += [f'--{key}', str(value)]
    if check_only:
        command.append('--check-only')
    return command

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--project', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    project = Path(args.project).resolve()
    cfg = load_config(args.config)
    if args.check_only:
        subprocess.run(make_command(project, cfg, cfg['seeds'][0], True), cwd=project, check=True)
        return
    output = resolve_path(project, cfg['output_dir'])
    if output.exists():
        raise FileExistsError(f'Choose a new output_dir in your config: {output}')
    output.mkdir(parents=True)
    # Snapshot experiment settings; each seed also saves its resolved CLI parameters.
    (output / 'experiment_config.json').write_text(json.dumps(cfg, indent=2))
    for seed in cfg['seeds']:
        subprocess.run(make_command(project, cfg, seed), cwd=project, check=True)


if __name__ == '__main__':
    main()
