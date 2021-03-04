import yaml
import argparse

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config_dir', type=str, default="configs/default_config.yaml",
                    help='Config File Location')
args = parser.parse_args()


with open("args.config_dir", 'r') as config_file:
    data = yaml.load(config_file)

print()