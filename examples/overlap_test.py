import os

from jax.config import config;

config.update("jax_enable_x64", True)

import multiprocessing
from tqdm import tqdm

import jax
import functools
import jax.numpy as np
import numpy as onp

from ff.handlers.deserialize import deserialize_handlers

from rdkit import Chem

from ff import handlers
from timemachine.potentials import bonded, shape
from timemachine.integrator import langevin_coefficients

from os.path import join

path_to_project = '/Users/jfass/Documents/GitHub/timemachine'
path_to_ligands = join(path_to_project, "tests/data/ligands_40.sdf")
path_to_forcefield = join(path_to_project, 'ff/params/smirnoff_1_1_0_ccc.py')

def recenter(conf):
    return conf - np.mean(conf, axis=0)


def get_conf(romol, idx):
    conformer = romol.GetConformer(idx)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    guest_conf /= 10
    return recenter(guest_conf)


def make_conformer(mol, conf_a, conf_b):
    mol.RemoveAllConformers()
    mol = Chem.CombineMols(mol, mol)
    cc = Chem.Conformer(mol.GetNumAtoms())
    conf = np.concatenate([conf_a, conf_b])
    conf *= 10
    for idx, pos in enumerate(onp.asarray(conf)):
        cc.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(cc)

    return mol


def get_heavy_atom_idxs(mol):
    idxs = []
    for a_idx, a in enumerate(mol.GetAtoms()):
        if a.GetAtomicNum() > 1:
            idxs.append(a_idx)
    return np.array(idxs, dtype=np.int32)


def convergence(args):
    epoch, lamb, lamb_idx = args

    suppl = Chem.SDMolSupplier(path_to_ligands, removeHs=False)

    ligands = []
    for mol in suppl:
        ligands.append(mol)

    ligand_a = ligands[0]
    ligand_b = ligands[1]

    # Use the following if you want to generate random molecules
    # ligand_a = Chem.AddHs(Chem.MolFromSmiles("CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"))
    # ligand_b = Chem.AddHs(Chem.MolFromSmiles("CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"))
    # ligand_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1CC"))
    # ligand_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1CC"))
    # AllChem.EmbedMolecule(ligand_a, randomSeed=2020)
    # AllChem.EmbedMolecule(ligand_b, randomSeed=2020)

    coords_a = get_conf(ligand_a, idx=0)
    coords_b = get_conf(ligand_b, idx=0)
    # uncomment if you want to apply a random rotation
    # coords_b = np.matmul(coords_b, special_ortho_group.rvs(3))

    coords_a = recenter(coords_a)
    coords_b = recenter(coords_b)

    coords = np.concatenate([coords_a, coords_b])

    # heavy atoms only
    a_idxs = get_heavy_atom_idxs(ligand_a)
    b_idxs = get_heavy_atom_idxs(ligand_b)

    # all atoms
    a_full_idxs = np.arange(0, ligand_a.GetNumAtoms())
    b_full_idxs = np.arange(0, ligand_b.GetNumAtoms())

    b_idxs += ligand_a.GetNumAtoms()
    b_full_idxs += ligand_a.GetNumAtoms()

    nrg_fns = []

    # load forcefield
    ff_raw = open(path_to_forcefield, "r").read()
    ff_handlers = deserialize_handlers(ff_raw)

    combined_mol = Chem.CombineMols(ligand_a, ligand_b)

    # parameterize only bonded and angle terms for speed
    for handler in ff_handlers:
        if isinstance(handler, handlers.HarmonicBondHandler):
            bond_idxs, (bond_params, _) = handler.parameterize(combined_mol)
            nrg_fns.append(
                functools.partial(bonded.harmonic_bond,
                                  params=bond_params,
                                  box=None,
                                  bond_idxs=bond_idxs
                                  )
            )
        elif isinstance(handler, handlers.HarmonicAngleHandler):
            angle_idxs, (angle_params, _) = handler.parameterize(combined_mol)
            nrg_fns.append(
                functools.partial(bonded.harmonic_angle,
                                  params=angle_params,
                                  box=None,
                                  angle_idxs=angle_idxs
                                  )
            )

    masses_a = onp.array([a.GetMass() for a in ligand_a.GetAtoms()]) * 10000
    masses_b = onp.array([a.GetMass() for a in ligand_b.GetAtoms()])

    combined_masses = np.concatenate([masses_a, masses_b])

    # center of mass restraint
    com_restraint_fn = functools.partial(bonded.centroid_restraint,
                                         params=None,
                                         box=None,
                                         lamb=None,
                                         masses=combined_masses,
                                         group_a_idxs=a_idxs,
                                         group_b_idxs=b_idxs,
                                         kb=50.0,
                                         b0=0.0)

    # set up shape parameters
    prefactor = 2.7  # unitless
    shape_lamb = (4 * np.pi) / (3 * prefactor)  # unitless
    kappa = np.pi / (np.power(shape_lamb, 2 / 3))  # unitless
    sigma = 0.15  # 1 angstrom std, 95% coverage by 2 angstroms
    alpha = kappa / (sigma * sigma)

    alphas = np.zeros(combined_mol.GetNumAtoms()) + alpha
    weights = np.zeros(combined_mol.GetNumAtoms()) + prefactor

    shape_restraint_fn = functools.partial(
        shape.harmonic_overlap,
        box=None,
        lamb=None,
        params=None,
        a_idxs=a_idxs,
        b_idxs=b_idxs,
        alphas=alphas,
        weights=weights,
        k=150.0
    )

    def restraint_fn(conf, lamb):
        return (1 - lamb) * com_restraint_fn(conf) + lamb * shape_restraint_fn(conf)

    nrg_fns.append(restraint_fn)

    def nrg_fn(conf, lamb):
        s = []
        for u in nrg_fns:
            s.append(u(conf, lamb=lamb))
        return np.sum(s)

    grad_fn = jax.grad(nrg_fn, argnums=(0, 1))
    grad_fn = jax.jit(grad_fn)

    du_dx_fn = jax.grad(nrg_fn, argnums=(0))
    du_dx_fn = jax.jit(du_dx_fn)

    x_t = coords
    v_t = np.zeros_like(x_t)

    w = Chem.SDWriter('frames_heavy_' + str(epoch) + '_' + str(lamb_idx) + '.sdf')

    dt = 1.5e-3
    ca, cb, cc = langevin_coefficients(300.0, dt, 1.0, combined_masses)
    cb = -1 * onp.expand_dims(cb, axis=-1)
    cc = onp.expand_dims(cc, axis=-1)

    du_dls = []

    # re-seed since forking preserves the seeded state
    onp.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    for step in tqdm(range(100000)):

        # uncomment if want to write out coordinates
        # if step % 1000 == 0:
        #     u = nrg_fn(x_t, lamb)
        #     print("step", step, "nrg", onp.asarray(u), "avg_du_dl",  onp.mean(du_dls))
        #     mol = make_conformer(combined_mol, x_t[:ligand_a.GetNumAtoms()], x_t[ligand_a.GetNumAtoms():])
        #     w.write(mol)
        #     w.flush()

        if step % 5 == 0 and step > 10000:
            du_dx, du_dl = grad_fn(x_t, lamb)
            du_dls.append(du_dl)
        else:
            du_dx = du_dx_fn(x_t, lamb)

        v_t = ca * v_t + cb * du_dx + cc * onp.random.normal(size=x_t.shape)
        x_t = x_t + v_t * dt

    return np.mean(onp.mean(du_dls))


if __name__ == "__main__":
    lambda_schedule = np.linspace(0, 1.0, )
    epoch = 0
    args = []
    for l_idx, lamb in enumerate(lambda_schedule):
        args.append((epoch, lamb, l_idx))

    # let's just look at a single "epoch" for the purposes of profiling
    avg_du_dls = convergence(args[0])
    avg_du_dls = np.asarray(avg_du_dls)
    print(avg_du_dls)
