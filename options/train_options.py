import argparse

class Train_ArgParser(argparse.ArgumentParser) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.formatter_class=argparse.ArgumentDefaultsHelpFormatter

        # Required Parameters
        required_args = self.add_argument_group('experiment information')
        required_args.add_argument('--name', type=str, help='job name', required=True)
        required_args.add_argument('--exp_dir', type=str, help='path of experiment directory', default='./result/')
        required_args.add_argument('-p', '--property', type=str, nargs='+', help='property list')

        # Model
        model_args = self.add_argument_group('model config')
        model_args.add_argument('--model_config', type=str, default='./config/model.yaml')

        # Data
        data_args = self.add_argument_group('data config')
        data_args.add_argument('--data_path', type=str, default='./data/ZINC/data.csv', help='data path')
        data_args.add_argument('--data_pkl_path', type=str, default='./data/ZINC/data.pkl', required=False, help='data pkl path(Optional)')
        data_args.add_argument('--library_path', type=str, default='./data/ZINC/library.csv', help='library path')
        data_args.add_argument('--split_path', type=str, default='./data/ZINC/split.csv', help='data split index path')

        # Hyperparameter
        hparams_args = self.add_argument_group('training hyperparameter config')
        hparams_args.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
        hparams_args.add_argument('--max_step', type=int, default=200000, help='Max Step')
        hparams_args.add_argument('--train_batch_size', type=int, default=512, help='Training Batch Size')
        hparams_args.add_argument('--val_batch_size', type=int, default=256, help='Validation Batch Size')
        hparams_args.add_argument('--num_validate', type=int, default=5, help='Number of Validation Iterations')
        hparams_args.add_argument('--condition_noise', type=float, default=0.0, help='Condition Noise')
        hparams_args.add_argument('--num_negative_samples', type=int, default=10, help='Hyperparameter for Negative Sampling')
        hparams_args.add_argument('--alpha', type=float, default=0.75, help='Hyperparameter for Negative Sampling')

        # Training Config
        train_args = self.add_argument_group('training config')
        train_args.add_argument('--gpus', type=int, default=1, help='Number of GPUS, only 0(cpu) or 1')
        train_args.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
        train_args.add_argument('--val_interval', type=int, default=2000, help='Valiation Interval')
        train_args.add_argument('--log_interval', type=int, default=1000, help='Logging Interval')
        train_args.add_argument('--print_interval', type=int, default=1000, help='Printing Interval')
        train_args.add_argument('--save_interval', type=int, default=10000, help='Model Checkpoint Interval')

