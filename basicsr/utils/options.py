import os
import re
import yaml
from collections import OrderedDict
from os import path as osp


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


_TEMPLATE_PATTERN = re.compile(r'\$\{([^}]+)\}')


def _get_by_dotted_key(data, dotted_key):
    current = data
    for key in dotted_key.split('.'):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def _resolve_string(value, full_opt):
    value = osp.expanduser(os.path.expandvars(value))

    def replace_fn(match):
        key = match.group(1).strip()
        if key == 'project_root':
            return full_opt['path']['root']
        resolved = _get_by_dotted_key(full_opt, key)
        if resolved is None or isinstance(resolved, (dict, list)):
            return match.group(0)
        return str(resolved)

    return _TEMPLATE_PATTERN.sub(replace_fn, value)


def _resolve_templates(value, full_opt, max_passes=10):
    current = value
    for _ in range(max_passes):
        resolved = _resolve_templates_once(current, full_opt)
        if resolved == current:
            break
        current = resolved
    return current


def _resolve_templates_once(value, full_opt):
    if isinstance(value, dict):
        return type(value)((k, _resolve_templates_once(v, full_opt))
                           for k, v in value.items())
    if isinstance(value, list):
        return [_resolve_templates_once(item, full_opt) for item in value]
    if isinstance(value, str):
        return _resolve_string(value, full_opt)
    return value


def _join_if_relative(path_value, base_dir):
    if path_value is None:
        return None
    path_value = osp.expanduser(os.path.expandvars(path_value))
    if base_dir and not osp.isabs(path_value):
        path_value = osp.join(base_dir, path_value)
    return osp.normpath(path_value)


def parse(opt_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train
    opt.setdefault('path', OrderedDict())
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    opt = _resolve_templates(opt, opt)

    # datasets
    if 'datasets' in opt:
        global_dataset_root = opt['path'].get('dataset_root')
        for phase, dataset in opt['datasets'].items():
            # for several datasets, e.g., test_1, test_2
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            if 'scale' in opt:
                dataset['scale'] = opt['scale']
            dataset_root = dataset.get('dataset_root', global_dataset_root)
            if dataset_root is not None:
                dataset['dataset_root'] = _join_if_relative(
                    dataset_root, opt['path']['root'])
            if dataset.get('dataroot_gt') is not None:
                dataset['dataroot_gt'] = _join_if_relative(
                    dataset['dataroot_gt'],
                    dataset.get('dataset_root', opt['path']['root']))
            if dataset.get('dataroot_lq') is not None:
                dataset['dataroot_lq'] = _join_if_relative(
                    dataset['dataroot_lq'],
                    dataset.get('dataset_root', opt['path']['root']))

    # paths
    for key, val in opt['path'].items():
        if isinstance(val, str) and key != 'root':
            if ('resume_state' in key or 'pretrain_network' in key
                    or key.endswith(('_root', '_dir', '_path', '_file'))):
                opt['path'][key] = _join_if_relative(val, opt['path']['root'])
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg
