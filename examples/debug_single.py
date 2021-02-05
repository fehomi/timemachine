# This script computes the relative binding free energy of a single edge.


import os
import argparse
import numpy as np
import jax

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem

from fe import topology
from fe import model
from md import builders

import functools

from ff import Forcefield
from ff.handlers.serialize import serialize_handlers
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers import nonbonded
from parallel.client import CUDAPoolClient


import multiprocessing

def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18


def wrap_method(args, fxn):
    # TODO: is there a more functools-y approach to make
    #   a function accept tuple instead of positional arguments?
    return fxn(*args)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Relative Binding Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        help="number of gpus",
        required=True
    )

    parser.add_argument(
        "--num_complex_windows",
        type=int,
        help="number of vacuum lambda windows",
        required=True
    )

    parser.add_argument(
        "--num_solvent_windows",
        type=int,
        help="number of solvent lambda windows",
        required=True
    )

    parser.add_argument(
        "--num_equil_steps",
        type=int,
        help="number of equilibration steps for each lambda window",
        required=True
    )

    parser.add_argument(
        "--num_prod_steps",
        type=int,
        help="number of production steps for each lambda window",
        required=True
    )


    cmd_args = parser.parse_args()
    multiprocessing.set_start_method('spawn')  # CUDA runtime is not forkable

    client = CUDAPoolClient(max_workers=cmd_args.num_gpus)

    # suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
    # all_mols = [x for x in suppl]
    # mol_a = all_mols[1]
    # mol_b = all_mols[4]

    mol_a = Chem.MolFromMolBlock("""
     RDKit          3D

 38 40  0  0  0  0  0  0  0  0999 V2000
   27.6174    1.5478  -11.1379 O   0  0  0  0  0  0  0  0  0  0  0  0
   27.1576    0.3016  -11.6549 C   0  0  0  0  0  0  0  0  0  0  0  0
   26.5229    0.3791  -13.0512 C   0  0  0  0  0  0  0  0  0  0  0  0
   25.0317    0.5852  -12.8454 C   0  0  0  0  0  0  0  0  0  0  0  0
   24.8099   -0.0815  -11.5012 C   0  0  0  0  0  0  0  0  0  0  0  0
   26.0230   -0.2460  -10.8113 C   0  0  0  0  0  0  0  0  0  0  0  0
   26.0229   -0.7786   -9.5060 C   0  0  0  0  0  0  0  0  0  0  0  0
   24.8037   -1.0879   -8.9007 C   0  0  0  0  0  0  0  0  0  0  0  0
   23.6089   -0.9192   -9.5831 C   0  0  0  0  0  0  0  0  0  0  0  0
   23.6054   -0.4269  -10.8974 C   0  0  0  0  0  0  0  0  0  0  0  0
   22.4927   -0.2591  -11.6812 O   0  0  0  0  0  0  0  0  0  0  0  0
   21.2585   -0.5430  -11.1867 C   0  0  0  0  0  0  0  0  0  0  0  0
   20.6891   -1.7973  -11.3987 C   0  0  0  0  0  0  0  0  0  0  0  0
   19.4288   -2.0656  -10.8661 C   0  0  0  0  0  0  0  0  0  0  0  0
   18.7466   -1.1093  -10.1294 C   0  0  0  0  0  0  0  0  0  0  0  0
   19.3039    0.1671   -9.9553 C   0  0  0  0  0  0  0  0  0  0  0  0
   18.5747    1.1946   -9.2462 C   0  0  0  0  0  0  0  0  0  0  0  0
   20.5607    0.4183  -10.4590 C   0  0  0  0  0  0  0  0  0  0  0  0
   27.5352   -1.0636   -8.5830 S   0  0  0  0  0  0  0  0  0  0  0  0
   27.2233   -2.0141   -7.4936 O   0  0  0  0  0  0  0  0  0  0  0  0
   28.5909   -1.3989   -9.5185 O   0  0  0  0  0  0  0  0  0  0  0  0
   27.9412    0.6140   -7.8295 C   0  0  0  0  0  0  0  0  0  0  0  0
   27.1179    0.8438   -6.8093 F   0  0  0  0  0  0  0  0  0  0  0  0
   18.0155    2.0188   -8.6719 N   0  0  0  0  0  0  0  0  0  0  0  0
   18.6828   -3.5869  -11.1124 Cl  0  0  0  0  0  0  0  0  0  0  0  0
   27.0640    1.3150  -13.8239 F   0  0  0  0  0  0  0  0  0  0  0  0
   26.6730   -0.7998  -13.6562 F   0  0  0  0  0  0  0  0  0  0  0  0
   28.3618    1.8681  -11.6713 H   0  0  0  0  0  0  0  0  0  0  0  0
   27.9828   -0.4090  -11.7192 H   0  0  0  0  0  0  0  0  0  0  0  0
   24.4208    0.1923  -13.6623 H   0  0  0  0  0  0  0  0  0  0  0  0
   24.7995    1.6472  -12.7681 H   0  0  0  0  0  0  0  0  0  0  0  0
   24.7791   -1.4880   -7.8979 H   0  0  0  0  0  0  0  0  0  0  0  0
   22.6949   -1.2250   -9.0991 H   0  0  0  0  0  0  0  0  0  0  0  0
   21.2136   -2.5719  -11.9366 H   0  0  0  0  0  0  0  0  0  0  0  0
   17.8151   -1.3988   -9.6937 H   0  0  0  0  0  0  0  0  0  0  0  0
   21.0174    1.3729  -10.2677 H   0  0  0  0  0  0  0  0  0  0  0  0
   28.9709    0.5746   -7.4730 H   0  0  0  0  0  0  0  0  0  0  0  0
   27.8363    1.3925   -8.5883 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1 28  1  0
  2  3  1  0
  2  6  1  0
  2 29  1  0
  3  4  1  0
  3 26  1  0
  3 27  1  0
  4  5  1  0
  4 30  1  0
  4 31  1  0
  5  6  2  0
  5 10  1  0
  6  7  1  0
  7  8  2  0
  7 19  1  0
  8  9  1  0
  8 32  1  0
  9 10  2  0
  9 33  1  0
 10 11  1  0
 11 12  1  0
 12 13  2  0
 12 18  1  0
 13 14  1  0
 13 34  1  0
 14 15  2  0
 14 25  1  0
 15 16  1  0
 15 35  1  0
 16 17  1  0
 16 18  2  0
 17 24  3  0
 18 36  1  0
 19 20  2  0
 19 21  2  0
 19 22  1  0
 22 23  1  0
 22 37  1  0
 22 38  1  0
M  END""", removeHs=False)

    mol_b = Chem.MolFromMolBlock("""
     RDKit          3D

 41 43  0  0  0  0  0  0  0  0999 V2000
   27.8905    0.7814  -11.2833 O   0  0  0  0  0  0  0  0  0  0  0  0
   26.9193   -0.0282  -11.8984 C   0  0  0  0  0  0  0  0  0  0  0  0
   26.2553    0.6485  -13.1070 C   0  0  0  0  0  0  0  0  0  0  0  0
   24.7743    0.3619  -13.0634 C   0  0  0  0  0  0  0  0  0  0  0  0
   24.5513   -0.0833  -11.6435 C   0  0  0  0  0  0  0  0  0  0  0  0
   25.7571   -0.3137  -10.9611 C   0  0  0  0  0  0  0  0  0  0  0  0
   25.7304   -0.7838   -9.6350 C   0  0  0  0  0  0  0  0  0  0  0  0
   24.4955   -0.9388   -8.9991 C   0  0  0  0  0  0  0  0  0  0  0  0
   23.3088   -0.7555   -9.6942 C   0  0  0  0  0  0  0  0  0  0  0  0
   23.3376   -0.3678  -11.0366 C   0  0  0  0  0  0  0  0  0  0  0  0
   22.2312   -0.2289  -11.8473 O   0  0  0  0  0  0  0  0  0  0  0  0
   21.0019   -0.5424  -11.3497 C   0  0  0  0  0  0  0  0  0  0  0  0
   20.5026   -1.8363  -11.4599 C   0  0  0  0  0  0  0  0  0  0  0  0
   19.3111   -2.1637  -10.8207 C   0  0  0  0  0  0  0  0  0  0  0  0
   18.8034   -3.4051  -10.9256 F   0  0  0  0  0  0  0  0  0  0  0  0
   18.6360   -1.2092  -10.0655 C   0  0  0  0  0  0  0  0  0  0  0  0
   19.1032    0.1048  -10.0073 C   0  0  0  0  0  0  0  0  0  0  0  0
   18.4198    1.1400   -9.2611 C   0  0  0  0  0  0  0  0  0  0  0  0
   20.2891    0.4299  -10.6551 C   0  0  0  0  0  0  0  0  0  0  0  0
   27.1782   -1.3347   -8.7177 S   0  0  0  0  0  0  0  0  0  0  0  0
   26.7390   -2.4320   -7.8613 O   0  0  0  0  0  0  0  0  0  0  0  0
   28.2843   -1.5524   -9.6512 O   0  0  0  0  0  0  0  0  0  0  0  0
   27.6153    0.0506   -7.5323 C   0  0  0  0  0  0  0  0  0  0  0  0
   26.3933    1.9664  -13.0190 F   0  0  0  0  0  0  0  0  0  0  0  0
   26.7651    0.2478  -14.2653 F   0  0  0  0  0  0  0  0  0  0  0  0
   17.8859    1.9702   -8.6637 N   0  0  0  0  0  0  0  0  0  0  0  0
   28.2838    1.1998   -8.2640 C   0  0  0  0  0  0  0  0  0  0  0  0
   28.4344    1.2213  -11.9476 H   0  0  0  0  0  0  0  0  0  0  0  0
   27.3623   -0.9733  -12.2077 H   0  0  0  0  0  0  0  0  0  0  0  0
   24.5370   -0.4389  -13.7581 H   0  0  0  0  0  0  0  0  0  0  0  0
   24.1815    1.2313  -13.3494 H   0  0  0  0  0  0  0  0  0  0  0  0
   24.4471   -1.2650   -7.9726 H   0  0  0  0  0  0  0  0  0  0  0  0
   22.3722   -0.9488   -9.1964 H   0  0  0  0  0  0  0  0  0  0  0  0
   21.0600   -2.5906  -11.9949 H   0  0  0  0  0  0  0  0  0  0  0  0
   17.7638   -1.5211   -9.5283 H   0  0  0  0  0  0  0  0  0  0  0  0
   20.6808    1.4261  -10.5650 H   0  0  0  0  0  0  0  0  0  0  0  0
   26.6995    0.3531   -7.0244 H   0  0  0  0  0  0  0  0  0  0  0  0
   28.2860   -0.3848   -6.7955 H   0  0  0  0  0  0  0  0  0  0  0  0
   28.6278    1.9639   -7.5645 H   0  0  0  0  0  0  0  0  0  0  0  0
   27.5819    1.6805   -8.9401 H   0  0  0  0  0  0  0  0  0  0  0  0
   29.1450    0.8679   -8.8468 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1 28  1  0
  2  3  1  0
  2  6  1  0
  2 29  1  0
  3  4  1  0
  3 24  1  0
  3 25  1  0
  4  5  1  0
  4 30  1  0
  4 31  1  0
  5  6  2  0
  5 10  1  0
  6  7  1  0
  7  8  2  0
  7 20  1  0
  8  9  1  0
  8 32  1  0
  9 10  2  0
  9 33  1  0
 10 11  1  0
 11 12  1  0
 12 13  2  0
 12 19  1  0
 13 14  1  0
 13 34  1  0
 14 15  1  0
 14 16  2  0
 16 17  1  0
 16 35  1  0
 17 18  1  0
 17 19  2  0
 18 26  3  0
 19 36  1  0
 20 21  2  0
 20 22  2  0
 20 23  1  0
 23 27  1  0
 23 37  1  0
 23 38  1  0
 27 39  1  0
 27 40  1  0
 27 41  1  0
M  END""", removeHs=False)

    # assert 0

    # (ytz): these are *binding* free energies, i.e. values that are less than zero.
    # label_dG_a = convert_uIC50_to_kJ_per_mole(float(mol_a.GetProp("IC50[uM](SPA)")))
    # label_dG_b = convert_uIC50_to_kJ_per_mole(float(mol_b.GetProp("IC50[uM](SPA)")))

    # print("binding dG_a", label_dG_a)
    # print("binding dG_b", label_dG_b)

    # label_ddG = label_dG_b - label_dG_a # complex - solvent
    label_ddG = 0

    print(mol_a.GetNumAtoms())
    print(mol_b.GetNumAtoms())

    # mol_a = 

    core = np.array([[ 1,  1],
       [ 2,  2],
       [ 3,  3],
       [ 4,  4],
       [ 5,  5],
       [ 6,  6],
       [ 7,  7],
       [ 8,  8],
       [ 9,  9],
       [10, 10],
       [11, 11],
       [12, 12],
       [13, 13],
       [14, 15],
       [15, 16],
       [16, 17],
       [17, 18],
       [18, 19],
       [20, 21],
       [23, 25],
       [24, 14],
       [31, 31],
       [32, 32],
       [33, 33],
       [34, 34],
       [35, 35]])

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    ff = Forcefield(ff_handlers)

    lambda_schedules = []

    for num_host_windows in [cmd_args.num_complex_windows, cmd_args.num_solvent_windows]:
        A = int(.35*num_host_windows)
        B = int(.30*num_host_windows)
        C = num_host_windows - A - B

        # Emprically, we see the largest variance in std <du/dl> near the endpoints in the nonbonded
        # terms. Bonded terms are roughly linear. So we add more lambda windows at the endpoint to
        # help improve convergence.
        lambda_schedule = np.concatenate([
            np.linspace(0.0,  0.25, A, endpoint=False),
            np.linspace(0.25, 0.75, B, endpoint=False),
            np.linspace(0.75, 1.0,  C, endpoint=True)
        ])

        assert len(lambda_schedule) == num_host_windows

        lambda_schedules.append(lambda_schedule)

    complex_schedule, solvent_schedule = lambda_schedules

    complex_schedule = np.concatenate([np.ones_like(complex_schedule), np.zeros_like(complex_schedule)])
    solvent_schedule = np.concatenate([np.ones_like(solvent_schedule), np.zeros_like(solvent_schedule)])

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system('tests/data/hif2a_nowater_min.pdb')
    complex_box += np.eye(3)*0.1 # BFGS this later

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)
    solvent_box += np.eye(3)*0.1 # BFGS this later

    # client = None

    binding_model = model.RBFEModel(
        client,
        ff,
        complex_system,
        complex_coords,
        complex_box,
        complex_schedule,
        solvent_system,
        solvent_coords,
        solvent_box,
        solvent_schedule,
        cmd_args.num_equil_steps,
        cmd_args.num_prod_steps
    )

    # vg_fn = jax.value_and_grad(binding_model.loss, argnums=0)

    ordered_params = ff.get_ordered_params()
    ordered_handles = ff.get_ordered_handles()

    gradient_clip_thresholds = {
        nonbonded.AM1CCCHandler: 0.05,
        nonbonded.LennardJonesHandler: np.array([0.001,0])
    }

    for epoch in range(100):

        epoch_params = serialize_handlers(ff_handlers)
        # loss, loss_grad = vg_fn(ordered_params, mol_a, mol_b, core, label_ddG)

        loss = binding_model.loss(ordered_params, mol_a, mol_b, core, label_ddG)

        assert 0

        print("epoch", epoch, "loss", loss)

        for loss_grad, handle in zip(loss_grad, ordered_handles):
            assert handle.params.shape == loss_grad.shape

            if type(handle) in gradient_clip_thresholds:
                bounds = gradient_clip_thresholds[type(handle)]
                loss_grad = np.clip(loss_grad, -bounds, bounds)
                print("adjust handle", handle, "by", loss_grad)
                handle.params -= loss_grad

                # useful for debugging to dump out the grads
                # for smirks, dp in zip(handle.smirks, loss_grad):
                    # if np.any(dp) > 0:
                        # print(smirks, dp)

        # write ff parameters after each epoch
        with open("checkpoint_"+str(epoch)+".py", 'w') as fh:
            fh.write(epoch_params)
