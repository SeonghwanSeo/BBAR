import argparse


class Train_ArgParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        # Experiment information
        exp_args = self.add_argument_group("experiment information")
        exp_args.add_argument("--name", type=str, help="job name", required=True)
        exp_args.add_argument("--exp_dir", type=str, help="path of experiment directory", default="./result/")
        exp_args.add_argument("-p", "--property", type=str, nargs="+", help="property list")

        # Model
        model_args = self.add_argument_group("model config")
        model_args.add_argument("--model_config", type=str, default="./config/model.yaml")

        # Data
        data_args = self.add_argument_group("data config")
        data_args.add_argument("--data_dir", type=str, default="./data/ZINC/", help="dataset directory")

        # Hyperparameter
        hparams_args = self.add_argument_group("training hyperparameter config")
        hparams_args.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
        hparams_args.add_argument("--max_step", type=int, default=200000, help="Max Step")
        hparams_args.add_argument("--train_batch_size", type=int, default=512, help="Training Batch Size")
        hparams_args.add_argument("--val_batch_size", type=int, default=256, help="Validation Batch Size")
        hparams_args.add_argument("--num_validate", type=int, default=5, help="Number of Validation Iterations")
        hparams_args.add_argument("--condition_noise", type=float, default=0.02, help="Condition Noise")
        hparams_args.add_argument(
            "--num_negative_samples", type=int, default=10, help="Hyperparameter for Negative Sampling"
        )
        hparams_args.add_argument("--alpha", type=float, default=0.75, help="Hyperparameter for Negative Sampling")

        hparams_args.add_argument("--lambda_term", type=float, default=1.0, help="Termination Loss Multiplier Factor")
        hparams_args.add_argument("--lambda_property", type=float, default=1.0, help="Property Loss Multiplier Factor")
        hparams_args.add_argument("--lambda_block", type=float, default=1.0, help="Block Loss Multiplier Factor")
        hparams_args.add_argument("--lambda_atom", type=float, default=1.0, help="Atom Loss Multiplier Factor")

        # Training Config
        train_args = self.add_argument_group("training config")
        train_args.add_argument("--gpus", type=int, default=1, help="Number of GPUS, only 0(cpu) or 1")
        train_args.add_argument("--num_workers", type=int, default=4, help="Number of Dataloader Workers")
        train_args.add_argument("--val_interval", type=int, default=2000, help="Valiation Interval(Step)")
        train_args.add_argument("--log_interval", type=int, default=100, help="Logging Interval(Stp)")
        train_args.add_argument("--print_interval", type=int, default=100, help="Printing Interval(Step)")
        train_args.add_argument("--save_interval", type=int, default=10000, help="Model Checkpoint Interval(Step)")
