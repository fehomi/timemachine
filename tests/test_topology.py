from jax.config import config; config.update("jax_enable_x64", True)

import unittest
import numpy as np

from fe import topology as topology

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

import jax


def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm


class BezenePhenolSparseTest(unittest.TestCase):

    def setUp(self, *args, **kwargs):

        suppl = Chem.SDMolSupplier('tests/data/benzene_phenol_sparse.sdf', removeHs=False)
        all_mols = [x for x in suppl]

        self.mol_a = all_mols[0]
        self.mol_b = all_mols[1]

        # atom type free
        ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_recharge.py').read())

        self.ff = Forcefield(ff_handlers)

        super(BezenePhenolSparseTest, self).__init__(*args, **kwargs)


    def test_bonded(self):
        # other bonded terms use an identical protocol, so we assume they're correct if the harmonic bond tests pass.

        # leaving benzene H unmapped, and phenol OH unmapped
        core = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
        ], dtype=np.int32)

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)

        (params_src, params_dst, params_uni), vjp_fn, (potential_src, potential_dst, potential_uni) = jax.vjp(st.parameterize_harmonic_bond, self.ff.hb_handle.params, has_aux=True)

        # test that vjp_fn works
        vjp_fn([np.random.rand(*params_src.shape), np.random.rand(*params_dst.shape), np.random.rand(*params_uni.shape)])

        assert len(potential_src.get_idxs() == 6)
        assert len(potential_dst.get_idxs() == 6)
        assert len(potential_uni.get_idxs() == 3)

        cc = self.ff.hb_handle.lookup_smirks("[#6X3:1]:[#6X3:2]")
        cH = self.ff.hb_handle.lookup_smirks("[#6X3:1]-[#1:2]")
        cO = self.ff.hb_handle.lookup_smirks("[#6X3:1]-[#8X2H1:2]")
        OH = self.ff.hb_handle.lookup_smirks("[#8:1]-[#1:2]")

        np.testing.assert_array_equal(params_src, [cc, cc, cc, cc, cc, cc])
        np.testing.assert_array_equal(params_dst, [cc, cc, cc, cc, cc, cc])
        np.testing.assert_array_equal(params_uni, [cH, cO, OH])


        core = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6]
        ], dtype=np.int32)

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)

        (params_src, params_dst, params_uni), vjp_fn, (potential_src, potential_dst, potential_uni) = jax.vjp(st.parameterize_harmonic_bond, self.ff.hb_handle.params, has_aux=True)

        assert len(potential_src.get_idxs() == 7)
        assert len(potential_dst.get_idxs() == 7)
        assert len(potential_uni.get_idxs() == 1)

        np.testing.assert_array_equal(params_src, [cc, cc, cc, cc, cc, cc, cH])
        np.testing.assert_array_equal(params_dst, [cc, cc, cc, cc, cc, cc, cO])
        np.testing.assert_array_equal(params_uni, [OH])

        # test that vjp_fn works
        vjp_fn([np.random.rand(*params_src.shape), np.random.rand(*params_dst.shape), np.random.rand(*params_uni.shape)])

    def test_nonbonded(self):

        # leaving benzene H unmapped, and phenol OH unmapped
        core = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
        ], dtype=np.int32)

        st = topology.SingleTopology(self.mol_a, self.mol_b, core, self.ff)
        x_a = get_romol_conf(self.mol_a)
        x_b = get_romol_conf(self.mol_b)

        # test interpolation of coordinates.
        x_src, x_dst = st.interpolate_params(x_a, x_b)
        x_avg = np.mean([x_src, x_dst], axis=0)

        assert x_avg.shape == (st.get_num_atoms(), 3)

        np.testing.assert_array_equal((x_a[:6] + x_b[:6])/2, x_avg[:6]) # C
        np.testing.assert_array_equal(x_a[6], x_avg[6]) # H
        np.testing.assert_array_equal(x_b[6:], x_avg[7:]) # OH

        params, vjp_fn, pot_c = jax.vjp(
            st.parameterize_nonbonded,
            self.ff.q_handle.params, 
            self.ff.lj_handle.params,
            has_aux=True
        )

        vjp_fn(np.random.rand(*params.shape))

        assert params.shape == (2*st.get_num_atoms(), 3) # qlj
        
        bt_a = topology.BaseTopology(self.mol_a, self.ff)
        qlj_a, pot_a = bt_a.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)
        bt_b = topology.BaseTopology(self.mol_b, self.ff)
        qlj_b, pot_b = bt_b.parameterize_nonbonded(self.ff.q_handle.params, self.ff.lj_handle.params)

        qlj_c = np.mean([params[:len(params)//2], params[len(params)//2:]], axis=0)

        np.testing.assert_array_equal((qlj_a[:6] + qlj_b[:6])/2, qlj_c[:6])
        np.testing.assert_array_equal(qlj_a[6], qlj_c[6]) # H
        np.testing.assert_array_equal(qlj_b[6:], qlj_c[7:]) # OH


class TestLigandSet(unittest.TestCase):

    def test_hif2a_ligands_dry_run(self):
        suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
        # test every combination in a dry run to ensure correctness
        all_mols = [x for x in suppl][:4]

        ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_recharge.py').read())

        ff = Forcefield(ff_handlers)

        # mcs_params = rdFMCS.MCSParameters()
        # mcs_params.AtomTyper = CompareDist()

        for mol_a in all_mols:
            for mol_b in all_mols:

                res = rdFMCS.FindMCS(
                    [mol_a, mol_b],
                )
                
                pattern = Chem.MolFromSmarts(res.smartsString)
                core_a = mol_a.GetSubstructMatch(pattern)
                core_b = mol_b.GetSubstructMatch(pattern)
                core = np.stack([core_a, core_b], axis=-1)

                st = topology.SingleTopology(mol_a, mol_b, core, ff)
                _ = jax.vjp(st.parameterize_harmonic_bond, ff.hb_handle.params, has_aux=True)
                _ = jax.vjp(st.parameterize_harmonic_angle, ff.ha_handle.params, has_aux=True)
                _ = jax.vjp(st.parameterize_proper_torsion, ff.pt_handle.params, has_aux=True)
                _ = jax.vjp(st.parameterize_improper_torsion, ff.it_handle.params, has_aux=True)
                _ = jax.vjp(st.parameterize_nonbonded, ff.q_handle.params, ff.lj_handle.params, has_aux=True)
