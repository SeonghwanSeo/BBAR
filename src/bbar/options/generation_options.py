import argparse


class Denovo_Generation_ArgParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        # Generator Parameters
        required_args = self.add_argument_group("required")
        required_args.add_argument(
            "-g", "--generator_config", type=str, default="./config/generator.yaml", help="generator config file"
        )

        # Optional Parameters
        opt_args = self.add_argument_group("optional")
        opt_args.add_argument("-o", "--output_path", type=str, help="output file name")
        opt_args.add_argument("-n", "--num_samples", type=int, help="number of generation", default=1)
        opt_args.add_argument("--seed", type=int, help="explicit random seed")
        opt_args.add_argument("-q", action="store_true", help="no print sampling script message")


class Scaffold_Generation_ArgParser(Denovo_Generation_ArgParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Scaffold-based Generation
        scaf_args = self.add_argument_group("scaffold-based generation")
        scaf_args.add_argument("-s", "--scaffold", type=str, default=None, help="scaffold SMILES")
        scaf_args.add_argument("-S", "--scaffold_path", type=str, default=None, help="scaffold SMILES path")
