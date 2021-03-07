import subprocess

#config_file = "configs/yaml/config_gradmatchpb_cifar10.yaml"
config_file = "configs/config_gradmatchpb_cifar10.py"
args = ['python']
args.append('train.py')
args.append('--config_file')
args.append(config_file)
subprocess.call(args)