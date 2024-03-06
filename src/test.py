import argparse
import yaml
from utils.trained_folder import EmsembleTrainedFolder, TrainedFolder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_config', type=str, help="test config file")
    _args = parser.parse_args()
    with open(_args.test_config, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)
    if len(configs['folder_name']) > 1:  # ensemble
        configs.update({'folder_name_list': configs['folder_name']})
        configs.pop('folder_name')
        tester = EmsembleTrainedFolder(**configs)
    else:
        configs.update({'folder_name': configs['folder_name'][0]})
        tester = TrainedFolder(**configs)
    tester.run_test()
