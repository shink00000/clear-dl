import argparse
import yaml

from cleardl.builder import (
    build_data_pipeline,
    build_framework,
    build_model,
    build_optimizer,
    build_scheduler,
    build_metrics
)
from cleardl.controller import Controller


def main(args):
    # build components
    with open(args.config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    Framework = build_framework(cfg['framework'])
    data_pipeline = build_data_pipeline(cfg['data_pipeline'])
    model = build_model(cfg['model'])
    optimizer = build_optimizer(cfg['optimizer'], named_parameters=model.named_parameters())
    scheduler = build_scheduler(cfg['scheduler'], optimizer=optimizer)
    metrics = build_metrics(cfg['metrics'])
    module = Framework(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics
    )

    # execute
    controller = Controller(
        module,
        data_pipeline,
        args.output_dir,
        args.max_epochs,
        args.eval_interval,
        args.resume_from
    )
    controller(args.mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'evaluate', 'inference'])
    parser.add_argument('config_path', type=str)
    parser.add_argument('--output_dir', type=str, default='./results/test')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()

    main(args)
