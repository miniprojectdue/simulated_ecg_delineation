#!/usr/bin/env python3
"""
common.py  -  paths, config loading, seeding and logging shared by every script here.

The repository root is found by walking up from this file until a folder containing
config/paths.yaml appears, which is the convention config/paths.yaml documents for
itself. Nothing in this package hardcodes an absolute path, so the same code runs
from a laptop, a session mount or a compute node without editing.
"""
import json
import os
import random
import sys
import time

import numpy as np

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEAD_INDEX = {name: i for i, name in enumerate(LEADS)}
N_LEADS = len(LEADS)

# Segmentation classes. Background is 0 so that an absent wave simply produces no region.
CLASS_NAMES = ['background', 'p', 'qrs', 't']
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
N_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = -100

# The eleven landmark columns, in canonical time order.
LANDMARKS = [
    'p_onset', 'p_peak', 'p_offset',
    'qrs_onset', 'q_peak', 'r_peak', 's_peak', 'qrs_offset',
    't_onset', 't_peak', 't_offset',
]
BOUNDARY_LANDMARKS = ['p_onset', 'p_offset', 'qrs_onset', 'qrs_offset', 't_onset', 't_offset']
PEAK_LANDMARKS = ['p_peak', 'q_peak', 'r_peak', 's_peak', 't_peak']

DISEASE_CLASSES = ['avblock', 'fam', 'iab', 'lae', 'lbbb', 'mi', 'rbbb', 'sinus']


def repo_root(start=None):
    """Walk up from this file until the folder holding config/paths.yaml is found."""
    here = os.path.abspath(start or __file__)
    if os.path.isfile(here):
        here = os.path.dirname(here)
    while True:
        if os.path.isfile(os.path.join(here, 'config', 'paths.yaml')):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise RuntimeError('could not locate the repository root, no config/paths.yaml found above %s' % start)
        here = parent


ROOT = repo_root()
ML_ROOT = os.path.join(ROOT, 'ml_modelling')


def ml_path(*parts):
    return os.path.join(ML_ROOT, *parts)


def load_yaml(path):
    """Load a YAML file, falling back to a small parser when PyYAML is absent."""
    try:
        import yaml
        with open(path) as fh:
            return yaml.safe_load(fh)
    except ImportError:
        return _mini_yaml(path)


def _mini_yaml(path):
    """A deliberately small reader for the flat two-level configs written here.

    It understands nested mappings by indentation, scalars, inline lists and the usual
    true/false/null spellings. It exists only so that a missing PyYAML cannot stop a run.
    """
    root = {}
    stack = [(-1, root)]
    with open(path) as fh:
        for raw in fh:
            line = raw.split('#')[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            key, _, value = line.strip().partition(':')
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            value = value.strip()
            if value == '':
                child = {}
                parent[key.strip()] = child
                stack.append((indent, child))
            else:
                parent[key.strip()] = _scalar(value)
    return root


def _scalar(text):
    if text.startswith('[') and text.endswith(']'):
        inner = text[1:-1].strip()
        return [] if not inner else [_scalar(p.strip()) for p in inner.split(',')]
    low = text.lower()
    if low in ('true', 'yes'):
        return True
    if low in ('false', 'no'):
        return False
    if low in ('null', 'none', '~'):
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def load_config(path, overrides=None, _seen=None):
    """Load a config, following any inherit key, then apply dotted command line overrides.

    A config may name a parent through inherit. The parent is loaded first and this file is
    merged on top of it, so a stage file only has to state what actually differs.
    """
    abs_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.isfile(abs_path):
        raise SystemExit('config not found at %s' % abs_path)
    _seen = _seen or []
    if abs_path in _seen:
        raise SystemExit('inherit cycle in the config chain %s' % (_seen + [abs_path]))
    cfg = load_yaml(abs_path) or {}
    parent_path = cfg.pop('inherit', None)
    if parent_path:
        parent = load_config(parent_path, overrides=None, _seen=_seen + [abs_path])
        cfg = deep_update(parent, cfg)
    cfg = apply_overrides(cfg, overrides)
    cfg.setdefault('_config_path', abs_path)
    return cfg


def deep_update(base, override):
    """Merge override into a copy of base, recursing into nested mappings."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def apply_overrides(cfg, pairs):
    """Apply a list of dotted key=value strings from the command line onto a config."""
    for pair in pairs or []:
        if '=' not in pair:
            raise SystemExit('override %r is not of the form section.key=value' % pair)
        dotted, _, value = pair.partition('=')
        node = cfg
        parts = dotted.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _scalar(value)
    return cfg


def set_seed(seed, deterministic=True):
    """Seed every generator the training path touches."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def pick_device(requested='auto'):
    import torch
    if requested and requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Tee(object):
    """Write to stdout and to a log file at the same time."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, 'a')
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.fh.write(text)

    def flush(self):
        self.stdout.flush()
        self.fh.flush()

    def close(self):
        self.fh.close()


def log(message):
    stamp = time.strftime('%H:%M:%S')
    print('[%s] %s' % (stamp, message), flush=True)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=2, default=_jsonable, sort_keys=True)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def environment_report():
    """Everything the Training Procedure section of the dissertation has to state."""
    report = {'python': sys.version.split()[0], 'numpy': np.__version__}
    try:
        import torch
        report['torch'] = torch.__version__
        report['cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report['cuda'] = torch.version.cuda
            report['gpu'] = torch.cuda.get_device_name(0)
    except ImportError:
        report['torch'] = 'not installed'
    try:
        import pandas
        report['pandas'] = pandas.__version__
    except ImportError:
        pass
    return report
