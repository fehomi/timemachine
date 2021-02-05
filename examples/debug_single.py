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

 39 41  0  0  0  0  0  0  0  0999 V2000
   27.4945    1.5901  -11.4326 O   0  0  0  0  0  0  0  0  0  0  0  0
   27.1374    0.2686  -11.7191 C   0  0  0  0  0  0  0  0  0  0  0  0
   26.4953    0.3323  -13.0981 C   0  0  0  0  0  0  0  0  0  0  0  0
   25.0217    0.5994  -12.8429 C   0  0  0  0  0  0  0  0  0  0  0  0
   24.8020   -0.0597  -11.4964 C   0  0  0  0  0  0  0  0  0  0  0  0
   26.0153   -0.2255  -10.8167 C   0  0  0  0  0  0  0  0  0  0  0  0
   26.0241   -0.7593   -9.5118 C   0  0  0  0  0  0  0  0  0  0  0  0
   24.8061   -1.0787   -8.9024 C   0  0  0  0  0  0  0  0  0  0  0  0
   23.6060   -0.9317   -9.5932 C   0  0  0  0  0  0  0  0  0  0  0  0
   23.6016   -0.4258  -10.8989 C   0  0  0  0  0  0  0  0  0  0  0  0
   22.4847   -0.2873  -11.6913 O   0  0  0  0  0  0  0  0  0  0  0  0
   21.2447   -0.5730  -11.1948 C   0  0  0  0  0  0  0  0  0  0  0  0
   20.5522    0.3978  -10.4720 C   0  0  0  0  0  0  0  0  0  0  0  0
   19.2931    0.1248   -9.9600 C   0  0  0  0  0  0  0  0  0  0  0  0
   18.6203    1.0830   -9.2845 F   0  0  0  0  0  0  0  0  0  0  0  0
   18.7413   -1.1375  -10.1417 C   0  0  0  0  0  0  0  0  0  0  0  0
   19.4210   -2.1088  -10.8684 C   0  0  0  0  0  0  0  0  0  0  0  0
   18.8530   -3.3145  -11.0601 F   0  0  0  0  0  0  0  0  0  0  0  0
   20.6751   -1.8286  -11.4013 C   0  0  0  0  0  0  0  0  0  0  0  0
   27.5391   -1.0647   -8.5878 S   0  0  0  0  0  0  0  0  0  0  0  0
   28.5975   -1.3882   -9.5392 O   0  0  0  0  0  0  0  0  0  0  0  0
   27.2312   -2.0153   -7.5145 O   0  0  0  0  0  0  0  0  0  0  0  0
   27.9407    0.4584   -7.8059 N   0  0  0  0  0  0  0  0  0  0  0  0
   27.0378    1.2658  -13.8907 F   0  0  0  0  0  0  0  0  0  0  0  0
   26.6284   -0.8376  -13.6973 F   0  0  0  0  0  0  0  0  0  0  0  0
   28.4928    1.5464   -8.6066 C   0  0  0  0  0  0  0  0  0  0  0  0
   28.3741    1.7859  -11.8038 H   0  0  0  0  0  0  0  0  0  0  0  0
   27.9892   -0.4124  -11.7122 H   0  0  0  0  0  0  0  0  0  0  0  0
   24.8113    1.6720  -12.7864 H   0  0  0  0  0  0  0  0  0  0  0  0
   24.3984    0.1783  -13.6254 H   0  0  0  0  0  0  0  0  0  0  0  0
   24.7828   -1.4635   -7.8934 H   0  0  0  0  0  0  0  0  0  0  0  0
   22.6875   -1.2371   -9.1179 H   0  0  0  0  0  0  0  0  0  0  0  0
   20.9875    1.3612  -10.2718 H   0  0  0  0  0  0  0  0  0  0  0  0
   17.7955   -1.3802   -9.7072 H   0  0  0  0  0  0  0  0  0  0  0  0
   21.2077   -2.5898  -11.9503 H   0  0  0  0  0  0  0  0  0  0  0  0
   28.3589    0.3734   -6.8803 H   0  0  0  0  0  0  0  0  0  0  0  0
   28.5484    2.4694   -8.0394 H   0  0  0  0  0  0  0  0  0  0  0  0
   27.7523    1.7872   -9.3272 H   0  0  0  0  0  0  0  0  0  0  0  0
   29.4721    1.2647   -9.0380 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1 27  1  0
  2  3  1  0
  2  6  1  0
  2 28  1  0
  3  4  1  0
  3 24  1  0
  3 25  1  0
  4  5  1  0
  4 29  1  0
  4 30  1  0
  5  6  2  0
  5 10  1  0
  6  7  1  0
  7  8  2  0
  7 20  1  0
  8  9  1  0
  8 31  1  0
  9 10  2  0
  9 32  1  0
 10 11  1  0
 11 12  1  0
 12 13  2  0
 12 19  1  0
 13 14  1  0
 13 33  1  0
 14 15  1  0
 14 16  2  0
 16 17  1  0
 16 34  1  0
 17 18  1  0
 17 19  2  0
 19 35  1  0
 20 21  2  0
 20 22  2  0
 20 23  1  0
 23 26  1  0
 23 36  1  0
 26 37  1  0
 26 38  1  0
 26 39  1  0
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

    core = np.array([[ 0,  0],
       [ 1,  1],
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
       [12, 18],
       [13, 16],
       [14, 15],
       [15, 13],
       [16, 14],
       [17, 12],
       [18, 19],
       [19, 21],
       [20, 20],
       [21, 22],
       [24, 17],
       [25, 23],
       [26, 24],
       [27, 26],
       [28, 27],
       [29, 29],
       [30, 28],
       [31, 30],
       [32, 31],
       [33, 34],
       [34, 33],
       [35, 32]])

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
