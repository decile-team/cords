import subprocess

config_file = "configs/yaml/config_gradmatchpb_cifar10.yaml"
args = ['python']
args.append('train.py')
args.append('--config_dir')
args.append(config_file)
subprocess.call(args)