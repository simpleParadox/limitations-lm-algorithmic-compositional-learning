from hydra import compose, initialize
from omegaconf import DictConfig
import argparse
from data_modules.synthetic_tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Choose config file")
    parser.add_argument("--config-name", type=str, required=True, help="Name of the config file (without .yaml)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize Hydra
    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name=args.config_name)
        for task_spec in cfg['data_mix']:
            print(f"\nGenerating data with config: {task_spec}")
            data_class_name = task_spec['task']
            excluded_keys = {'task', 'how_many', 'logname'}
            param_dict = {k: v for k, v in task_spec.items() if k not in excluded_keys}
            data_obj = eval(data_class_name)(**param_dict)
            # data = data_obj.generate(task_spec['how_many'])
            data = data_obj.generate(10)
            print("First data point:")
            print(data[0])

if __name__ == "__main__":
    main()