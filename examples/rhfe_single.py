# relative hydration free energy with a single topology protocol.
import os
import argparse
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem

from fe import topology
from md import builders
from md import minimizer

import functools

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers


import multiprocessing

from fe import free_energy


def wrap_method(args, fn):
    gpu_idx = args[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    return fn(*args[1:])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Relative Hydration Free Energy Consistency Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        help="number of gpus"
    )

    parser.add_argument(
        "--num_vacuum_windows",
        type=int,
        help="number of vacuum lambda windows"
    )

    parser.add_argument(
        "--num_solvent_windows",
        type=int,
        help="number of solvent lambda windows"
    )

    parser.add_argument(
        "--num_equil_steps",
        type=int,
        help="number of equilibration lambda windows"
    )

    parser.add_argument(
        "--num_prod_steps",
        type=int,
        help="number of production lambda windows"
    )

    parser.add_argument(
        "--num_absolute_windows",
        type=int,
        help="number of absolute lambda windows"
    )

    cmd_args = parser.parse_args()

    multiprocessing.set_start_method('spawn') # CUDA runtime is not forkable
    pool = multiprocessing.Pool(cmd_args.num_gpus)

    suppl = Chem.SDMolSupplier('tests/data/benzene_flourinated.sdf', removeHs=False)
    all_mols = [x for x in suppl]

    mol_a = all_mols[0]
    mol_b = all_mols[1]

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    ff = Forcefield(ff_handlers)

    # the water system first.
    solvent_system, solvent_coords, solvent_box, omm_topology = builders.build_water_system(4.0)
    solvent_box += np.eye(3)*0.1 # BFGS this later

    print("Minimizing the host structure to remove clashes.")
    minimized_solvent_coords = minimizer.minimize_host_4d(mol_a, solvent_system, solvent_coords, ff, solvent_box)

    absolute_lambda_schedule = np.concatenate([
        np.linspace(0.0, 0.333, cmd_args.num_absolute_windows - cmd_args.num_absolute_windows//3, endpoint=False),
        np.linspace(0.333, 1.0, cmd_args.num_absolute_windows//3),
    ])

    abs_dGs = []

    for idx, mol in enumerate([mol_a, mol_b]):

        afe = free_energy.AbsoluteFreeEnergy(mol, ff)
        absolute_args = []

        for lambda_idx, lamb in enumerate(absolute_lambda_schedule):
            gpu_idx = lambda_idx % cmd_args.num_gpus
            absolute_args.append((gpu_idx, lamb, solvent_system, minimized_solvent_coords, solvent_box, cmd_args.num_equil_steps, cmd_args.num_prod_steps))

        results = pool.map(functools.partial(wrap_method, fn=afe.host_edge), absolute_args, chunksize=1)

        for lamb, (bonded_du_dl, nonbonded_du_dl) in zip(absolute_lambda_schedule, results):
            print("final absolute", idx ,"lambda", lamb, "bonded:", bonded_du_dl, "nonbonded:", nonbonded_du_dl)

        dG = np.trapz([x[0]+x[1] for x in results], absolute_lambda_schedule)
        print("mol", idx, "dG absolute:", dG)
        abs_dGs.append(dG)

    print("Absolute Difference", abs_dGs[0] - abs_dGs[1])

    # relative free energy, compare two different core approaches

    core_full = np.stack([
        np.arange(mol_a.GetNumAtoms()),
        np.arange(mol_b.GetNumAtoms())
    ], axis=1)

    core_part = np.stack([
        np.arange(mol_a.GetNumAtoms() - 1),
        np.arange(mol_b.GetNumAtoms() - 1)
    ], axis=1)

    # test relative free energy:

    # full mapping
    # partial mapping



    for core_idx, core in enumerate([core_full, core_part]):

        rfe = free_energy.RelativeFreeEnergy(mol_a, mol_b, core, ff)

        vacuum_lambda_schedule = np.linspace(0.0, 1.0, cmd_args.num_vacuum_windows)
        solvent_lambda_schedule = np.linspace(0.0, 1.0, cmd_args.num_solvent_windows)

        # vacuum leg
        vacuum_args = []
        for lambda_idx, lamb in enumerate(vacuum_lambda_schedule):
            gpu_idx = lambda_idx % cmd_args.num_gpus
            vacuum_args.append((gpu_idx, lamb, cmd_args.num_equil_steps, cmd_args.num_prod_steps))

        results = pool.map(functools.partial(wrap_method, fn=rfe.vacuum_edge), vacuum_args, chunksize=1)

        for lamb, (bonded_du_dl, nonbonded_du_dl) in zip(vacuum_lambda_schedule, results):
            print("final vacuum lambda", lamb, "bonded:", bonded_du_dl, "nonbonded:", nonbonded_du_dl)

        dG_vacuum = np.trapz([x[0]+x[1] for x in results], vacuum_lambda_schedule)
        print("dG vacuum:", dG_vacuum)

        solvent_args = []
        for lambda_idx, lamb in enumerate(solvent_lambda_schedule):
            gpu_idx = lambda_idx % cmd_args.num_gpus
            solvent_args.append((gpu_idx, lamb, solvent_system, minimized_solvent_coords, solvent_box, cmd_args.num_equil_steps, cmd_args.num_prod_steps))
        
        results = pool.map(functools.partial(wrap_method, fn=rfe.host_edge), solvent_args, chunksize=1)

        for lamb, (bonded_du_dl, nonbonded_du_dl) in zip(solvent_lambda_schedule, results):
            print("final solvent lambda", lamb, "bonded:", bonded_du_dl, "nonbonded:", nonbonded_du_dl)

        dG_solvent = np.trapz([x[0]+x[1] for x in results], solvent_lambda_schedule)
        print("dG solvent:", dG_solvent)

        print("Core", core_idx, "Difference", dG_solvent - dG_vacuum)

