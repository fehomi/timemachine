import numpy as np

# forcefield handlers
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

# free energy classes
from fe.free_energy import RelativeFreeEnergy, construct_lambda_schedule
from fe.model import RBFEModel

# MD initialization
from md import builders

# parallelization across multiple GPUs
from parallel.client import CUDAPoolClient

from collections import namedtuple

from pathlib import Path

from pickle import load
from typing import List

from time import time

# how much MD to run, on how many GPUs
Configuration = namedtuple(
    'Configuration',
    ['num_gpus', 'num_complex_windows', 'num_solvent_windows', 'num_equil_steps', 'num_prod_steps'])

configuration = Configuration(
    num_gpus=10, num_complex_windows=60, num_solvent_windows=60, num_equil_steps=10000, num_prod_steps=100000)

# locations relative to project root
root = Path(__file__).absolute().parent.parent.parent
path_to_protein = str(root.joinpath('tests/data/hif2a_nowater_min.pdb'))

# locations relative to example folder
path_to_results = Path(__file__).absolute().parent
path_to_transformations = str(path_to_results.joinpath('relative_transformations.pkl'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generating predictions using a snapshot force field",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--path_to_ff",
        type=str,
        help="path to forcefield .py file",
        default=str(root.joinpath('ff/params/smirnoff_1_1_0_ccc.py'))
    )

    parser.add_argument(
        "--save_path",
        type=str,
        help="path to save predictions .npy file",
        default=str(root.joinpath('examples/hif2a/predictions.npy'))
    )

    cmd_args = parser.parse_args()

    # load and construct forcefield
    with open(cmd_args.path_to_ff) as f:
        ff_handlers = deserialize_handlers(f.read())
    forcefield = Forcefield(ff_handlers)
    ordered_params = forcefield.get_ordered_params()

    # set up multi-GPU client
    client = CUDAPoolClient(max_workers=configuration.num_gpus)

    # load pre-defined collection of relative transformations
    with open(path_to_transformations, 'rb') as f:
        relative_transformations: List[RelativeFreeEnergy] = load(f)

    # build the complex system
    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(path_to_protein)
    # TODO: optimize box
    complex_box += np.eye(3) * 0.1  # BFGS this later

    # build the water system
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
    # TODO: optimize box
    solvent_box += np.eye(3) * 0.1  # BFGS this later

    # note: "complex" means "protein + solvent"
    binding_model = RBFEModel(
        client=client,
        ff=forcefield,
        complex_system=complex_system,
        complex_coords=complex_coords,
        complex_box=complex_box,
        complex_schedule=construct_lambda_schedule(configuration.num_complex_windows),
        solvent_system=solvent_system,
        solvent_coords=solvent_coords,
        solvent_box=solvent_box,
        solvent_schedule=construct_lambda_schedule(configuration.num_solvent_windows),
        equil_steps=configuration.num_equil_steps,
        prod_steps=configuration.num_prod_steps,
    )

    predictions = []
    for rfe in relative_transformations:
        # compute a prediction, measuring total wall-time
        t0 = time()

        # TODO: save du_dl so we can assess prediction uncertainty
        predictions.append(binding_model.predict(ordered_params, rfe.mol_a, rfe.mol_b, rfe.core))

        t1 = time()
        elapsed = t1 - t0

        print(f'completed prediction in {elapsed:.3f} s !')
        print(f'\tprediction: {predictions[-1]:.3f}\n\tlabel: {rfe.label:.3f}')

        np.save(cmd_args.save_path, predictions)
