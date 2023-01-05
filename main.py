import argparse
import yaml
from train_utils.trainer import PoseNDF_trainer
import shutil


def train(config, config_path):
    trainer = PoseNDF_trainer(config)
    copy_config = f"{config['experiment']['root_dir']}/{trainer.exp_name}/config.yaml"
    shutil.copyfile(config_path, copy_config)
    val = config['experiment'].get('val', False)
    test = config['experiment'].get('test', False)
    if test:
        trainer.inference(trainer.ep)
    for i in range(trainer.ep, config['train']['max_epoch']):
        print(f"Starting epoch {i}")
        _, _ = trainer.train_model(i)
        if val and i % 2 == 0:
            trainer.validate(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train PoseNDF.'
    )
    parser.add_argument('--config', '-c', default='configs/amass.yaml', type=str, help='Path to config file.')
    parser.add_argument('--test', '-t', action="store_true")
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    train(config, config_path)
