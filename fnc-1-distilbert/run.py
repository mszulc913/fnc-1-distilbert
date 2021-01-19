import argparse
import logging
import sys

import transformers

from train import train_val, train_test

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(log_format)
logger.addHandler(handler)


def main():
    parser = argparse.ArgumentParser(description='Train DistilBERT for FNC data.')
    parser.add_argument(
        'mode', type=str, help=f'Training type. Must be one of: ["train-test", "train-val"].'
                               f'Run "train-test" to train the model on the whole train '
                               f'set and evaluate it on test set.'
                               f'Run "train-val" to train the model on 80%% of the train '
                               f'set and evaluate it on the remaining 20%%.')
    parser.add_argument('--lr', help='Learning rate.', type=float, default=5e-5)
    parser.add_argument(
        '--use-scheduler', action='store_true',
        help='Whether to use learning rate scheduler and decrease the learning rate by half after 10 epochs.')
    parser.add_argument('--epochs', help='Number of training epochs.', type=int, default=10)
    parser.add_argument('--use-class-weights', help='Whether to weight the loss function.', action='store_true')
    parser.add_argument('--seed', help='Random seed.', type=int, default=None)
    parser.add_argument('--weight-decay', help='Adam optimizer weights decay rate.', type=float, default=0)
    parser.add_argument(
        '--name', help='Name of the experiment that will be placed in results folder name.',
        type=str, default=None)

    args = parser.parse_args()

    transformers.set_seed(args.seed)

    if args.mode == 'train-test':
        train_test(
            n_epochs=args.epochs,
            weight_decay=args.weight_decay,
            use_class_weights=args.use_class_weights,
            learning_rate=args.lr,
            experiment_name=args.name,
            use_scheduler=args.use_scheduler
        )
    else:
        train_val(
            seed=args.seed,
            n_epochs=args.epochs,
            weight_decay=args.weight_decay,
            use_class_weights=args.use_class_weights,
            learning_rate=args.lr,
            experiment_name=args.name,
            use_scheduler=args.use_scheduler
        )


if __name__ == "__main__":
    main()
