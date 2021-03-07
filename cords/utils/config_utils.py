import os
import os.path as osp
from pathlib import Path
import ast
import yaml
import importlib.util
import copy
import os


def is_str(x):
    """Whether the input is an string instance.
    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def is_filepath(x):
    return is_str(x) or isinstance(x, Path)


def fopen(filepath, *args, **kwargs):
    if is_str(filepath):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)
    raise ValueError('`filepath` should be a string or a Path')


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    else:
        dir_name = osp.expanduser(dir_name)
        os.makedirs(dir_name, mode=mode, exist_ok=True)


def _validate_py_syntax(filename):
    with open(filename, 'r') as f:
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError('There are syntax errors in config '
                          f'file {filename}: {e}')


def load_config_data(filepath):
    filename = osp.abspath(osp.expanduser(filepath))
    check_file_exist(filename)
    fileExtname = osp.splitext(filename)[1]
    if fileExtname not in ['.py', '.yaml', '.yml']:
        raise IOError('Only py/yml/yaml type are supported now!')
    """
    Parsing Config file
    """
    if filename.endswith('.yaml'):
        with open(filename, 'r') as config_file:
            configdata = yaml.load(config_file, Loader=yaml.FullLoader)
    elif filename.endswith('.py'):
        spec = importlib.util.spec_from_file_location("config", filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        configdata = copy.deepcopy(mod.config)
    return configdata