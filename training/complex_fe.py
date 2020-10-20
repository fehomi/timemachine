import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)

import argparse
import time
import datetime
import numpy as np
import os
import sys
import copy

from fe import standard_state

from ff import handlers
from ff.handlers.serialize import serialize_handlers
from ff.handlers.deserialize import deserialize_handlers

from rdkit import Chem

import configparser
import grpc

from training import dataset
from training import model, setup_system
from training import simulation
from training import service_pb2_grpc

from timemachine.potentials import jax_utils
from timemachine.lib import LangevinIntegrator, potentials
from training import build_system

from simtk import unit
from simtk.openmm import app

# from fe import PDBWriter

# used during visualization to bring everything back to home box
def recenter(conf, box):

    new_coords = []

    periodicBoxSize = box

    for atom in conf:
        diff = np.array([0., 0., 0.])
        diff += periodicBoxSize[2]*np.floor(atom[2]/periodicBoxSize[2][2]);
        diff += periodicBoxSize[1]*np.floor((atom[1]-diff[1])/periodicBoxSize[1][1]);
        diff += periodicBoxSize[0]*np.floor((atom[0]-diff[0])/periodicBoxSize[0][0]);
        new_coords.append(atom - diff)

    return np.array(new_coords)

def add_restraints(combined_coords, ligand_idxs, pocket_idxs, temperature):

    restr_k = 100.0 # force constant for the restraint
    restr_avg_xi = np.mean(combined_coords[ligand_idxs], axis=0)
    restr_avg_xj = np.mean(combined_coords[pocket_idxs], axis=0)
    restr_ctr_dij = np.sqrt(np.sum((restr_avg_xi - restr_avg_xj)**2))

    restr = potentials.CentroidRestraint(
        np.array(ligand_idxs, dtype=np.int32),
        np.array(pocket_idxs, dtype=np.int32),
        masses,
        restr_k,
        restr_ctr_dij,
        precision=np.float32
    )


    ssc = standard_state.harmonic_com_ssc(
        restr_k,
        restr_ctr_dij,
        temperature
    )

    return restr, ssc

def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18

# (ytz): need to add box to this
def find_protein_pocket_atoms(conf, box, nha, nwa, search_radius):
    """
    Find atoms in the protein that are close to the binding pocket. This simply grabs the
    protein atoms that are within search_radius nm of each ligand atom.

    The ordering of the atoms in the conformation should be:

    |nha|nwa|nla|

    Where, nha is the number of protein atoms, nwa is the number of water atoms, and nla
    is the number of ligand atoms.

    Parameters
    ----------
    conf: np.array [N,3]
        system coordinates

    box: np.array [N,3]
        periodic box

    nha: int
        number of host atoms

    nwa: int
        number of water atoms

    search_radius: float
        how far we search into the binding pocket.

    """
    # (ytz): this is horribly slow and can be made much faster
    rl = np.expand_dims(conf[nha+nwa:], axis=1) # ligand atoms
    rp = np.expand_dims(conf[:nha], axis=0) # protein_atoms

    # rl = np.expand_dims(rl, axis=0)
    # rp = np.expand_dims(rp, axis=1)
    # (YTZ) ADD PBC SUPPORT FOR RESTRAINT FINDING
    diff = rl - rp

    # apply PBCs
    # for d in range(3):
        # diff -= box[d]*np.floor(np.expand_dims(diff[...,d], axis=-1)/box[d][d]+0.5)

    # ligand x protein
    dij = np.sqrt(np.sum(np.power(diff, 2), axis=-1))

    pocket_atoms = set()

    for dists in dij:
        for p_idx, d in enumerate(dists):
            # print(d, search_radius)
            if d < search_radius:
                pocket_atoms.add(p_idx)

    # print("pocket_atoms", pocket_atoms)

    return list(pocket_atoms)
        # nns = np.argsort(dists)
        # for p_idx in nns:
            # if dists[p_idx] < search_radius

    assert 0

    ri = np.expand_dims(conf, axis=0)
    rj = np.expand_dims(conf, axis=1)
    dij = jax_utils.distance(ri, rj)


    # pdd = dij[:(nha+nwa), :(nha+nwa)]
    # pdd = pdd + (np.eye(nha+nwa)*10000)
    # print("SHORTEST DISTANCE", np.amin(pdd))



    pocket_atoms = set()

    for l_idx, dists in enumerate(dij[nha+nwa:]):
        nns = np.argsort(dists[:nha])
        for p_idx in nns:
            if dists[p_idx] < search_radius:
                pocket_atoms.add(p_idx)

    return list(pocket_atoms)

# def setup_restraints(
    # ligand_idxs,
    # pocket_idxs,
    # combined):


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Absolute Hydration Free Energy Script')
    parser.add_argument('--config_file', type=str, required=True, help='Location of config file.')
    
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)
    print("Config Settings:")
    config.write(sys.stdout)

    general_cfg = config['general']


    # basic gist of workflow:
    # 1. configure learning rates for the optimizer
    # 2. load freesolv dataset from SDF file
    # 3. split dataset into train/test
    # 4. connect to workers
    # 5. deserialize off smirnoff parameters
    # 6. prepare water box
    # 7. for each epoch, first run on test set then shuffled training set
    # 8. save parameters after each molecule

    # set up learning rates
    learning_rates = {}
    for k, v in config['learning_rates'].items():
        vals = [float(x) for x in v.split(',')]
        if k == 'am1ccc':
            learning_rates[handlers.AM1CCCHandler] = np.array(vals)
        elif k == 'lj':
            learning_rates[handlers.LennardJonesHandler] = np.array(vals)

    intg_cfg = config['integrator']

    suppl = Chem.SDMolSupplier(general_cfg['ligand_sdf'], removeHs=False)

    data = []

    for guest_idx, mol in enumerate(suppl):
        # label_dG = -4.184*float(mol.GetProp(general_cfg['bind_prop'])) # in kcal/mol
        # label_dG = -1*convert_uIC50_to_kJ_per_mole(float(mol.GetProp(general_cfg['bind_prop'])))
        label_dG = -1*float(mol.GetProp(general_cfg['bind_prop']))*4.18
        # label_err = 4.184*float(mol.GetProp(general_cfg['dG_err'])) # errs are positive!
        # label_dG = 80
        label_err = 0
        data.append((mol, label_dG, label_err))

    full_dataset = dataset.Dataset(data)
    np.random.seed(2020)
    print("shuffling dataset")
    full_dataset.shuffle() # random split

    train_frac = float(general_cfg['train_frac'])
    train_dataset, test_dataset = full_dataset.split(train_frac)

    forcefield = general_cfg['forcefield']

    stubs = []

    ff_raw = open(forcefield, "r").read()
    ff_handlers = deserialize_handlers(ff_raw)

    protein_system, protein_coords, nwa, nha, protein_box = build_system.build_protein_system(general_cfg['protein_pdb'])

    water_system, water_coords, water_box = build_system.build_water_system(box_width=3.0)

    num_steps = int(general_cfg['n_steps'])

    raw_schedules = config['lambda_schedule']
    schedules = {}
    for k, v in raw_schedules.items():
        schedules[k] = np.array([float(x) for x in v.split(',')])

    # move this to model class
    worker_address_list = []
    for address in config['workers']['hosts'].split(','):
        worker_address_list.append(address)

    for address in worker_address_list:
        # print("connecting to", address)
        channel = grpc.insecure_channel(address,
            options = [
                ('grpc.max_send_message_length', 500 * 1024 * 1024),
                ('grpc.max_receive_message_length', 500 * 1024 * 1024)
            ]
        )

        stub = service_pb2_grpc.WorkerStub(channel)
        stubs.append(stub)

    for epoch in range(100):

        print("Starting Epoch", epoch, datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        epoch_dir = os.path.join(general_cfg["out_dir"], "epoch_"+str(epoch))

        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        epoch_params = serialize_handlers(ff_handlers)
        with open(os.path.join(epoch_dir, "start_epoch_params.py"), 'w') as fh:
            fh.write(epoch_params)

        all_data = []

        # fix m
        test_items = [(x, True) for x in test_dataset.data]
        # (ytz): re-enable me
        train_dataset.shuffle()
        train_items = [(x, False) for x in train_dataset.data]

        all_data.extend(test_items)
        all_data.extend(train_items)

        # debug
        # all_data = all_data[:1]

        reorg_dG = float(general_cfg['reorg_dG'])

        for idx, ((mol, label_dG, label_err), inference) in enumerate(all_data):

            # combined_pdb = Chem.CombineMols(
                # Chem.MolFromPDBFile("debug_solvated.pdb", removeHs=False),
                # mol
            # )
            # combined_pdb_str = Chem.MolToPDBBlock(combined_pdb)
            # with open("holy_debug.pdb", "w") as fh:
                # fh.write(combined_pdb_str

            if inference:
                prefix = "test"
            else:
                prefix = "train"

            ligand_idxs = np.arange(mol.GetNumAtoms()) + nha + nwa

            start_time = time.time()

            # restraints
            # combined_potentials.append(restraint)
            # vjp_fns.append([])

            # seed = np.random.randint(0, np.iinfo(np.int32).max)
            seed = 0 # zero seed will let worker randomize it.

            stage_dGs = []
            # stage_vjp_fns = []
            # stage_grads = []

            handle_and_grads = {}

            min_lamb_start = 0.5

            for stage in [0,1,2]:

                if stage == 0:
                    # out_dir = os.path.join(epoch_dir, "mol_"+mol.GetProp("_Name"))\
                    # if not os.path.exists(out_dir):
                        # os.makedirs(out_dir)

                    # safety guard
                    # try:

                    guest_lambda_offset_idxs = np.ones(mol.GetNumAtoms(), dtype=np.int32) 

                    combined_potentials, masses, vjp_fns = setup_system.combine_potentials(
                        ff_handlers,
                        mol,
                        water_system,
                        guest_lambda_offset_idxs,
                        precision=np.float32
                    )

                    combined_coords = setup_system.combine_coordinates(
                        water_coords,
                        mol
                    )

                    lambda_schedule = schedules['solvent']

                    simulation_box = water_box

                    min_lamb_end = None

                    nha_count = water_coords.shape[0]

                if stage == 1:

                    guest_lambda_offset_idxs = np.zeros(mol.GetNumAtoms(), dtype=np.int32) 

                    combined_potentials, masses, vjp_fns = setup_system.combine_potentials(
                        ff_handlers,
                        mol,
                        protein_system,
                        guest_lambda_offset_idxs,
                        precision=np.float32
                    )

                    combined_coords = setup_system.combine_coordinates(
                        protein_coords,
                        mol
                    )

                    pocket_idxs = find_protein_pocket_atoms(combined_coords, protein_box, nha, nwa, 0.4)

                    restraint_potential, ssc = add_restraints(
                        combined_coords,
                        ligand_idxs,
                        pocket_idxs,
                        float(intg_cfg['temperature'])
                    )
                    restr = potentials.LambdaPotential(restraint_potential, len(masses), 0).bind(np.array([]))
                    combined_potentials.append(restr)
                    vjp_fns.append([])

                    lambda_schedule = schedules['complex_restraints']
                    simulation_box = protein_box

                    min_lamb_end = 0.0

                    nha_count = nha + nwa

                if stage == 2:

                    guest_lambda_offset_idxs = np.ones(mol.GetNumAtoms(), dtype=np.int32) 

                    combined_potentials, masses, vjp_fns = setup_system.combine_potentials(
                        ff_handlers,
                        mol,
                        protein_system,
                        guest_lambda_offset_idxs,
                        precision=np.float32
                    )

                    combined_coords = setup_system.combine_coordinates(
                        protein_coords,
                        mol
                    )

                    pocket_idxs = find_protein_pocket_atoms(combined_coords, protein_box, nha, nwa, 0.4)

                    restraint_potential, ssc = add_restraints(
                        combined_coords,
                        ligand_idxs,
                        pocket_idxs,
                        float(intg_cfg['temperature'])
                    )
                    combined_potentials.append(restraint_potential.bind(np.array([])))
                    vjp_fns.append([])

                    lambda_schedule = schedules['complex_decouple']
                    simulation_box = protein_box

                    min_lamb_end = None

                    nha_count = nha + nwa

                intg = LangevinIntegrator(
                    float(intg_cfg['temperature']),
                    float(intg_cfg['dt']),
                    float(intg_cfg['friction']),
                    masses,
                    seed
                )

                # tbd fix me and check boundary errors
                # simulation_box = simulation_box + np.eye(3) + 0.2

                sim = simulation.Simulation(
                    combined_coords,
                    np.zeros_like(combined_coords),
                    simulation_box,
                    combined_potentials,
                    intg
                )

                du_dls, grad_dG = model.simulate(
                    sim,
                    num_steps,
                    lambda_schedule,
                    nha_count,
                    min_lamb_start,
                    min_lamb_end,
                    stubs
                )

                dG = np.trapz(du_dls, lambda_schedule)
                stage_dGs.append(dG)

                for grad, handle_and_vjp_fns in zip(grad_dG, vjp_fns):
                    for handle, vjp_fn in handle_and_vjp_fns:
                        dp = vjp_fn(grad)[0]
                        if handle not in handle_and_grads:
                            handle_and_grads[handle] = dp
                        else:
                            handle_and_grads[handle] += dp

                # stage_grads.append(grad_dG)
                # stage_vjp_fns.append(vjp_fns)

                print(stage, dG)

                plt.plot(lambda_schedule, du_dls)
                plt.ylabel("du_dlambda")
                plt.xlabel("lambda")
                plt.savefig(os.path.join(epoch_dir, str(stage)+"_ti_mol_"+mol.GetProp("_Name")))
                plt.clf()

            print(stage_dGs, ssc)
            pred_dG = np.sum(stage_dGs) - ssc - reorg_dG

            loss = np.abs(pred_dG - label_dG)

            # (ytz) bootstrap CI on TI is super janky
            # error CIs are wrong "95% CI [{:.2f}, {:.2f}, {:.2f}]".format(pred_err.lower_bound, pred_err.value, pred_err.upper_bound),
            print("epoch", epoch, prefix, "mol", mol.GetProp("_Name"), "loss {:.2f}".format(loss), "pred_dG {:.2f}".format(pred_dG), "label_dG {:.2f}".format(label_dG), "label err {:.2f}".format(label_err), "time {:.2f}".format(time.time() - start_time), "smiles:", Chem.MolToSmiles(mol))

            # update ff parameters
            if not inference:

                loss_grad = np.sign(pred_dG - label_dG)
                # assert len(grad_dG) == len(vjp_fns)

                for handle, grad in handle_and_grads.items():
                    if type(handle) in learning_rates:
                        bounds = learning_rates[type(handle)]

                        dL_dp = loss_grad * grad
                        dL_dp = np.clip(dL_dp, -bounds, bounds)

                        handle.params -= dL_dp

                # for grad, handle_and_vjp_fns in zip(grad_dG, vjp_fns):
                #     for handle, vjp_fn in handle_and_vjp_fns:
                #         if type(handle) in learning_rates:

                #             bounds = learning_rates[type(handle)]
                #             dL_dp = loss_grad*vjp_fn(grad)[0]
                #             dL_dp = np.clip(dL_dp, -bounds, bounds)

                #             handle.params -= dL_dp

                epoch_params = serialize_handlers(ff_handlers)

                # write parameters after each traning molecule
                with open(os.path.join(epoch_dir, "checkpoint_params_idx_"+str(idx)+"_mol_"+mol.GetProp("_Name")+".py"), 'w') as fh:
                    fh.write(epoch_params)

            # assert 0
            # except Exception as e:
            #     import traceback
            #     print("Exception in mol", mol.GetProp("_Name"), Chem.MolToSmiles(mol), e)
            #     traceback.print_exc()


        # epoch_params = serialize_handlers(ff_handlers)
        # with open(os.path.join(epoch_dir, "end_epoch_params.py"), 'w') as fh:
        #     fh.write(epoch_params)