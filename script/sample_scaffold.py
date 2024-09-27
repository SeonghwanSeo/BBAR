import random
import time
from omegaconf import OmegaConf
from rdkit import Chem

from utils.seed import set_seed

from bbar.generate import MoleculeBuilder
from bbar.options.generation_options import Scaffold_Generation_ArgParser


def setup_generator():
    # Parsing
    parser = Scaffold_Generation_ArgParser()
    args, remain_args = parser.parse_known_args()
    generator_cfg = OmegaConf.load(args.generator_config)

    # Overwrite Config
    if args.model_path is not None:
        generator_cfg.model_path = args.model_path
    if args.library_path is not None:
        generator_cfg.library_path = args.library_path
    if args.library_builtin_model_path is not None:
        generator_cfg.library_builtin_model_path = args.library_builtin_model_path
    generator = MoleculeBuilder(generator_cfg)

    # Second Parsing To Read Condition
    if len(generator.target_properties) > 0:
        for property_name in generator.target_properties:
            parser.add_argument(f"--{property_name}", type=float, required=True)
    args = parser.parse_args()
    condition = {property_name: args.__dict__[property_name] for property_name in generator.target_properties}

    return generator, args, condition


def main():
    # Set Generator
    generator, args, condition = setup_generator()

    # Set Output File
    if args.output_path not in [None, "null"]:
        output_path = args.output_path
    else:
        output_path = "/dev/null"
    out_writer = open(output_path, "w")

    # Load Scaffold
    assert (args.scaffold_path is not None) or (args.scaffold is not None), "No Scaffold!"
    if args.scaffold_path is not None:
        with open(args.scaffold_path) as f:
            scaffold_list = [l.strip() for l in f.readlines()]
    else:
        scaffold_list = [args.scaffold]

    # Set Seed
    if args.seed is None:
        args.seed = random.randint(0, 1e6)

    # Start
    global_st = time.time()
    global_success = 0
    for scaf_idx, scaffold_smi in enumerate(scaffold_list):
        print(f"[{scaf_idx+1}/{len(scaffold_list)}] Scaffold: {scaffold_smi}")
        scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
        if scaffold_mol is None:
            print("Fail to RDKit Encoding")
            continue

        local_st = time.time()
        success = 0
        for i in range(args.num_samples):
            seed = args.seed + i

            set_seed(seed)
            generated_mol = generator.generate(scaffold_mol, condition)
            if generated_mol is not None:
                generated_smiles = Chem.MolToSmiles(generated_mol)
            else:
                generated_smiles = None

            if generated_smiles is None:
                if not args.q:
                    print(f"[{i+1}(Seed {seed})] FAIL")
            else:
                if not args.q:
                    print(f"[{i+1}(Seed {seed})] {generated_smiles}")
                out_writer.write(generated_smiles + "\n")
                success += 1

        local_end = time.time()
        time_cost = local_end - local_st
        global_success += success
        print(f"Num Generated Mol: {success}")
        print(f"Time Cost: {time_cost:.3f}, {time_cost/args.num_samples:.3f}\n")

    out_writer.close()
    global_end = time.time()

    if len(scaffold_list) > 1:
        time_cost = global_end - global_st
        print(f"Num Generated Mol: {global_success}")
        print(f"Total Time Cost: {time_cost:.3f}, {time_cost/args.num_samples/len(scaffold_list):.3f}")


if __name__ == "__main__":
    main()
