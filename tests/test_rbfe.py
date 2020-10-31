# test protocols for setting up relative binding free energy calculations.

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.lib import potentials, custom_ops
from timemachine.lib import LangevinIntegrator

from ff.handlers import openmm_deserializer
from ff.handlers.deserialize import deserialize_handlers

from fe import rbfe
from md import Recipe
from md import builders

def test_stage_0():

    benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    phenol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1"))

    AllChem.EmbedMolecule(benzene)
    AllChem.EmbedMolecule(phenol)

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    r_benzene = Recipe.from_rdkit(benzene, ff_handlers)
    r_phenol = Recipe.from_rdkit(phenol, ff_handlers)

    r_combined = r_benzene.combine(r_phenol)

    core_pairs = np.array([
        [0,1],
        [1,2],
        [2,3],
        [3,4],
        [4,5]
    ], dtype=np.int32)
    core_pairs[:, 1] += benzene.GetNumAtoms()

    b_idxs = np.arange(phenol.GetNumAtoms()) + benzene.GetNumAtoms()

    com_k = 10.0
    core_k = 200.0
    rbfe.stage_0(r_combined, b_idxs, core_pairs, com_k, core_k)

    centroid_count = 0
    core_count = 0
    nb_count = 0

    for bp in r_combined.bound_potentials:
        if isinstance(bp, potentials.LambdaPotential):
            u_fn = bp.get_u_fn()
            if isinstance(u_fn, potentials.CentroidRestraint):
                centroid_count += 1

                # (1-lambda)*u_fn
                np.testing.assert_equal(bp.get_multiplier(), -1.0)
                np.testing.assert_equal(bp.get_offset(), 1.0)

                np.testing.assert_equal(u_fn.get_a_idxs(), core_pairs[:, 0])
                np.testing.assert_equal(u_fn.get_b_idxs(), core_pairs[:, 1])

            elif isinstance(u_fn, potentials.CoreRestraint):
                core_count += 1

                # lambda*u_fn
                np.testing.assert_equal(bp.get_multiplier(), 1.0)
                np.testing.assert_equal(bp.get_offset(), 0.0)

                np.testing.assert_equal(u_fn.get_bond_idxs(), core_pairs)

        elif isinstance(bp, potentials.Nonbonded):
            nb_count += 1
            test_plane_idxs = bp.get_lambda_plane_idxs()
            ref_plane_idxs = np.zeros_like(test_plane_idxs)
            ref_plane_idxs[benzene.GetNumAtoms():] = 1
            np.testing.assert_array_equal(ref_plane_idxs, test_plane_idxs)

            test_offset_idxs = bp.get_lambda_offset_idxs()
            np.testing.assert_array_equal(test_offset_idxs, np.zeros_like(test_offset_idxs))

        # test C++ side of things
        bp.bound_impl(precision=np.float32)

    assert nb_count == 1
    assert centroid_count == 1
    assert core_count == 1

def test_stage_1():

    benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    phenol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1"))

    AllChem.EmbedMolecule(benzene)
    AllChem.EmbedMolecule(phenol)

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    r_benzene = Recipe.from_rdkit(benzene, ff_handlers)
    r_phenol = Recipe.from_rdkit(phenol, ff_handlers)

    r_combined = r_benzene.combine(r_phenol)

    core_pairs = np.array([
        [0,1],
        [1,2],
        [2,3],
        [3,4],
        [4,5]
    ], dtype=np.int32)
    core_pairs[:, 1] += benzene.GetNumAtoms()

    a_idxs = np.arange(benzene.GetNumAtoms())
    b_idxs = np.arange(phenol.GetNumAtoms()) + benzene.GetNumAtoms()

    com_k = 10.0
    core_k = 200.0
    rbfe.stage_1(r_combined, a_idxs, b_idxs, core_pairs, core_k)

    core_count = 0
    nb_count = 0
    for bp in r_combined.bound_potentials:
        if isinstance(bp, potentials.LambdaPotential):
            assert 0
        elif isinstance(bp, potentials.CentroidRestraint):
            assert 0
        elif isinstance(bp, potentials.CoreRestraint):
            core_count += 1
            np.testing.assert_equal(bp.get_bond_idxs(), core_pairs)
        elif isinstance(bp, potentials.Nonbonded):

            nb_count += 1

            test_plane_idxs = bp.get_lambda_plane_idxs()
            np.testing.assert_array_equal(test_plane_idxs, np.zeros_like(test_plane_idxs))

            test_offset_idxs = bp.get_lambda_offset_idxs()
            ref_offset_idxs = np.zeros_like(test_offset_idxs)
            ref_offset_idxs[benzene.GetNumAtoms():] = 1
            np.testing.assert_array_equal(ref_offset_idxs, test_offset_idxs)

            # ensure exclusions are added correctly
            combined_idxs = bp.get_exclusion_idxs()

            left_idxs = r_benzene.bound_potentials[-1].get_exclusion_idxs()
            right_idxs = r_phenol.bound_potentials[-1].get_exclusion_idxs()

            n_left = benzene.GetNumAtoms()
            n_right = phenol.GetNumAtoms()

            assert (len(left_idxs) + len(right_idxs) + n_left * n_right) == len(combined_idxs)
            test_set = np.sort(combined_idxs[len(left_idxs) + len(right_idxs):])

            ref_set = []
            for i in range(n_left):
                for j in range(n_right):
                    ref_set.append((i, j+n_left))
            ref_set = np.sort(np.array(ref_set, dtype=np.int32))

            np.testing.assert_array_equal(test_set, ref_set)

        bp.bound_impl(precision=np.float32)

    assert nb_count == 1
    assert core_count == 1

def get_romol_conf(mol):
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    guest_conf = guest_conf/10 # from angstroms to nm
    return np.array(guest_conf, dtype=np.float64)

def test_water_system_stage_0():

    benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1")) # a
    phenol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1")) # b

    AllChem.EmbedMolecule(benzene)
    AllChem.EmbedMolecule(phenol)

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    r_benzene = Recipe.from_rdkit(benzene, ff_handlers)
    r_phenol = Recipe.from_rdkit(phenol, ff_handlers)

    r_combined = r_benzene.combine(r_phenol)

    core_pairs = np.array([
        [0,1],
        [1,2],
        [2,3],
        [3,4],
        [4,5]
    ], dtype=np.int32)
    core_pairs[:, 1] += benzene.GetNumAtoms()

    b_idxs = np.arange(phenol.GetNumAtoms()) + benzene.GetNumAtoms()

    com_k = 10.0
    core_k = 200.0
    rbfe.stage_0(r_combined, b_idxs, core_pairs, com_k, core_k)

    system, host_coords, box, topology = builders.build_water_system(5.0)

    r_host = Recipe.from_openmm(system)
    r_final = r_host.combine(r_combined)

    # minimize coordinates of host + ligand A
    ha_coords = np.concatenate([
        host_coords,
        get_romol_conf(benzene)
    ])

    ha_coords = rbfe.minimize(r_host, r_benzene, ha_coords, box)

    x0 = np.concatenate([
        ha_coords,
        get_romol_conf(phenol)
    ])

    # production run at various values of lambda
    avg_du_dls = []
    for lamb in [0.0, 0.5, 1.0]:
        print("production run with lamb", lamb)
        u_impls = []
        for bp in r_final.bound_potentials:
            # print(bp)
            u_impls.append(bp.bound_impl(precision=np.float32))

        seed = np.random.randint(np.iinfo(np.int32).max)

        masses = np.concatenate([r_host.masses, r_benzene.masses, r_phenol.masses])

        intg = LangevinIntegrator(
            300.0,
            1.5e-3,
            1.0,
            masses,
            seed
        ).impl()

        v0 = np.zeros_like(x0)

        ctxt = custom_ops.Context(
            x0,
            v0,
            box,
            intg,
            u_impls
        )

        # equilibration
        for lamb in range(10000):
            ctxt.step(lamb)

        du_dl_obs = custom_ops.AvgPartialUPartialLambda(u_impls, 25)
        ctxt.add_observable(du_dl_obs)

        # add observable for <du/dl>
        for lamb in range(10000):
            ctxt.step(lamb)

        avg_du_dls.append(du_dl_obs.avg_du_dl())

        assert np.any(np.abs(ctxt.get_x_t()) > 100) == False
        assert np.any(np.isnan(ctxt.get_x_t())) == False
        assert np.any(np.isinf(ctxt.get_x_t())) == False

    # should be monotonically decreasing
    assert avg_du_dls[0] > avg_du_dls[1]
    assert avg_du_dls[1] > avg_du_dls[2]