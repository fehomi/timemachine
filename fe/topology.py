import numpy as np
import jax
import jax.numpy as jnp

from timemachine.lib import potentials
from ff.handlers import nonbonded, bonded

_SCALE_12 = 1.0
_SCALE_13 = 1.0
_SCALE_14 = 0.5
_BETA = 2.0
_CUTOFF = 1.2

_MAX_PERIOD = 6
_MAX_PHASE = 4

class HostGuestTopology():

    def __init__(self, host_p, guest_topology):
        """
        Utility tool for combining host with a guest, in that order.

        Parameters
        ----------
        host_p:
            Nonbonded potential for the host.

        guest_topology:
            Guest's Topology {Base, Dual, Single}Topology.

        """
        self.guest_topology = guest_topology
        self.host_p = host_p
        self.num_host_atoms = len(self.host_p.get_lambda_plane_idxs())

    def get_num_atoms(self):
        return self.num_host_atoms + self.guest_topology.get_num_atoms()

    def parameterize_harmonic_bond(self, ff_params):
        params, potential = self.guest_topology.parameterize_harmonic_bond(ff_params)
        potential.set_bond_idxs(potential.get_bond_idxs() + self.num_host_atoms)
        return params, potential

    def parameterize_harmonic_angle(self, ff_params):
        params, potential = self.guest_topology.parameterize_harmonic_angle(ff_params)
        potential.set_angle_idxs(potential.get_angle_idxs() + self.num_host_atoms)
        return params, potential

    def parameterize_proper_torsion(self, ff_params):
        params, potential = self.guest_topology.parameterize_proper_torsion(ff_params)
        potential.set_torsion_idxs(potential.get_torsion_idxs() + self.num_host_atoms)
        return params, potential

    def parameterize_improper_torsion(self, ff_params):
        params, potential = self.guest_topology.parameterize_improper_torsion(ff_params)
        potential.set_torsion_idxs(potential.get_torsion_idxs() + self.num_host_atoms)
        return params, potential

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        # this needs to take care of the case when there's parameter interpolation.
        num_guest_atoms = self.guest_topology.get_num_atoms()
        guest_qlj, guest_p = self.guest_topology.parameterize_nonbonded(ff_q_params, ff_lj_params)
        
        # see if we're doing parameter interpolation
        assert guest_qlj.shape[1] == 3

        assert guest_p.get_beta() == self.host_p.get_beta()
        assert guest_p.get_cutoff() == self.host_p.get_cutoff()

        hg_exclusion_idxs = np.concatenate([self.host_p.get_exclusion_idxs(), guest_p.get_exclusion_idxs() + self.num_host_atoms])
        hg_scale_factors = np.concatenate([self.host_p.get_scale_factors(), guest_p.get_scale_factors()])
        hg_lambda_offset_idxs = np.concatenate([self.host_p.get_lambda_offset_idxs(), guest_p.get_lambda_offset_idxs()])
        hg_lambda_plane_idxs = np.concatenate([self.host_p.get_lambda_plane_idxs(), guest_p.get_lambda_plane_idxs()])

        if guest_qlj.shape[0] == num_guest_atoms:
            # no parameter interpolation
            hg_nb_params = jnp.concatenate([self.host_p.params, guest_qlj])

            return hg_nb_params, potentials.Nonbonded(
                hg_exclusion_idxs,
                hg_scale_factors,
                hg_lambda_plane_idxs,
                hg_lambda_offset_idxs,
                guest_p.get_beta(),
                guest_p.get_cutoff()
            )

        elif guest_qlj.shape[0] == num_guest_atoms*2:
            # with parameter interpolation
            hg_nb_params_src = jnp.concatenate([self.host_p.params, guest_qlj[:num_guest_atoms]])
            hg_nb_params_dst = jnp.concatenate([self.host_p.params, guest_qlj[num_guest_atoms:]])
            hg_nb_params = jnp.concatenate([hg_nb_params_src, hg_nb_params_dst])

            nb = potentials.Nonbonded(
                hg_exclusion_idxs,
                hg_scale_factors,
                hg_lambda_plane_idxs,
                hg_lambda_offset_idxs,
                guest_p.get_beta(),
                guest_p.get_cutoff()
            )

            return hg_nb_params, potentials.InterpolatedPotential(nb, self.get_num_atoms(), hg_nb_params.size)

        else:
            # you dun' goofed and consequences will never be the same
            assert 0


class BaseTopology():

    def __init__(self, mol, forcefield):
        """
        Utility for working with a single ligand.

        Parameter
        ---------
        mol: ROMol
            Ligand to be parameterized

        forcefield: ff.Forcefield
            A convenience wrapper for forcefield lists.

        """
        self.mol = mol
        self.ff = forcefield

    def get_num_atoms(self):
        return self.mol.GetNumAtoms()

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        q_params = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol)
        lj_params = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol)

        exclusion_idxs, scale_factors = nonbonded.generate_exclusion_idxs(
            self.mol,
            scale12=_SCALE_12,
            scale13=_SCALE_13,
            scale14=_SCALE_14
        )

        scale_factors = np.stack([scale_factors, scale_factors], axis=1)

        N = len(q_params)

        lambda_plane_idxs = np.zeros(N, dtype=np.int32)
        lambda_offset_idxs = np.ones(N, dtype=np.int32)

        beta = _BETA
        cutoff = _CUTOFF # solve for this analytically later

        nb = potentials.Nonbonded(
            exclusion_idxs,
            scale_factors,
            lambda_plane_idxs,
            lambda_offset_idxs,
            beta,
            cutoff
        ) 

        params = jnp.concatenate([
            jnp.reshape(q_params, (-1, 1)),
            jnp.reshape(lj_params, (-1, 2))
        ], axis=1)

        return params, nb

    def parameterize_harmonic_bond(self, ff_params):
        params, idxs = self.ff.hb_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.HarmonicBond(idxs)


    def parameterize_harmonic_angle(self, ff_params):
        params, idxs = self.ff.ha_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.HarmonicAngle(idxs)


    def parameterize_proper_torsion(self, ff_params):
        params, idxs = self.ff.pt_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.PeriodicTorsion(idxs)


    def parameterize_improper_torsion(self, ff_params):
        params, idxs = self.ff.it_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.PeriodicTorsion(idxs)



class DualTopology():

    def __init__(self, mol_a, mol_b, forcefield):
        """
        Utility for working with two ligands via dual topology. Both copies of the ligand
        will be present after merging.

        Parameter
        ---------
        mol_a: ROMol
            First ligand to be parameterized

        mol_b: ROMol
            Second ligand to be parameterized

        forcefield: ff.Forcefield
            A convenience wrapper for forcefield lists.

        """
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.ff = forcefield

    def get_num_atoms(self):
        return self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms()

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        q_params_a = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_a)
        q_params_b = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_b) # HARD TYPO
        lj_params_a = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_a)
        lj_params_b = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_b)

        q_params = jnp.concatenate([q_params_a, q_params_b])
        lj_params = jnp.concatenate([lj_params_a, lj_params_b])

        exclusion_idxs_a, scale_factors_a = nonbonded.generate_exclusion_idxs(
            self.mol_a,
            scale12=_SCALE_12,
            scale13=_SCALE_13,
            scale14=_SCALE_14
        )

        exclusion_idxs_b, scale_factors_b = nonbonded.generate_exclusion_idxs(
            self.mol_b,
            scale12=_SCALE_12,
            scale13=_SCALE_13,
            scale14=_SCALE_14
        )

        mutual_exclusions = []
        mutual_scale_factors = []

        NA = self.mol_a.GetNumAtoms()
        NB = self.mol_b.GetNumAtoms()

        for i in range(NA):
            for j in range(NB):
                mutual_exclusions.append([i, j + NA])
                mutual_scale_factors.append([1.0, 1.0])

        mutual_exclusions = np.array(mutual_exclusions)
        mutual_scale_factors = np.array(mutual_scale_factors)

        combined_exclusion_idxs = np.concatenate([
            exclusion_idxs_a,
            exclusion_idxs_b + NA,
            mutual_exclusions
        ]).astype(np.int32)

        combined_scale_factors = np.concatenate([
            np.stack([scale_factors_a, scale_factors_a], axis=1),
            np.stack([scale_factors_b, scale_factors_b], axis=1),
            mutual_scale_factors
        ]).astype(np.float64)

        combined_lambda_plane_idxs = np.zeros(NA+NB, dtype=np.int32)
        combined_lambda_offset_idxs = np.concatenate([
            np.ones(NA, dtype=np.int32),
            np.ones(NB, dtype=np.int32)
        ])

        beta = _BETA
        cutoff = _CUTOFF # solve for this analytically later

        nb = potentials.Nonbonded(
            combined_exclusion_idxs,
            combined_scale_factors,
            combined_lambda_plane_idxs,
            combined_lambda_offset_idxs,
            beta,
            cutoff
        ) 

        params = jnp.concatenate([
            jnp.reshape(q_params, (-1, 1)),
            jnp.reshape(lj_params, (-1, 2))
        ], axis=1)

        return params, nb

    def _parameterize_bonded_term(self, ff_params, bonded_handle, potential):
        offset = self.mol_a.GetNumAtoms()
        params_a, idxs_a = bonded_handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = bonded_handle.partial_parameterize(ff_params, self.mol_b)
        params_c = jnp.concatenate([params_a, params_b])
        idxs_c = jnp.concatenate([idxs_a, idxs_b + offset])
        return params_c, potential(idxs_c)

    def parameterize_harmonic_bond(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.hb_handle, potentials.HarmonicBond)


    def parameterize_harmonic_angle(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.ha_handle, potentials.HarmonicAngle)


    def parameterize_proper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.pt_handle, potentials.PeriodicTorsion)


    def parameterize_improper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.it_handle, potentials.PeriodicTorsion)


class SingleTopology():

    def __init__(self, mol_a, mol_b, core, ff):
        """
        SingleTopology combines two molecules through a common core.
        """
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.ff = ff

        # map into idxs in the combined molecule
        self.a_to_c = np.arange(mol_a.GetNumAtoms(), dtype=np.int32) # identity
        self.b_to_c = np.zeros(mol_b.GetNumAtoms(), dtype=np.int32) - 1

        self.NC = mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - len(core)

        # mark membership:
        # 0: Core
        # 1: R_A (default)
        # 2: R_B
        self.c_flags = np.ones(self.get_num_atoms(), dtype=np.int32)

        for a, b in core:
            self.c_flags[a] = 0
            self.b_to_c[b] = a

        iota = self.mol_a.GetNumAtoms()
        for b_idx, c_idx in enumerate(self.b_to_c):
            if c_idx == -1:
                self.b_to_c[b_idx] = iota
                self.c_flags[iota] = 2
                iota += 1

        # test for uniqueness
        # (ytz): fix me for np arrays

        # print(core[:, 0])
        # print(tuple(core[:, 0]))

        # assert len(set(tuple(core[:, 0]))) == len(core[:, 0])
        # assert len(set(tuple(core[:, 1]))) == len(core[:, 1])

    def get_num_atoms(self):
        return self.NC

    def interpolate_params(self, params_a, params_b):
        """
        Interpolate two sets of per-particle parameters.

        This can be used to interpolate nonbonded parameters,
        coordinates, etc.

        Parameters
        ----------
        params_a: np.ndarray, shape [N_A, ...]
            Parameters for the mol_a

        params_b: np.ndarray, shape [N_B, ...]
            Parameters for the mol_b

        Returns
        -------
        tuple: (src, dst)
            Two np.ndarrays each of shape [N_C, ...]

        """

        src_params = [None]*self.get_num_atoms()
        dst_params = [None]*self.get_num_atoms()

        for a_idx, c_idx in enumerate(self.a_to_c):
            src_params[c_idx] = params_a[a_idx]
            dst_params[c_idx] = params_a[a_idx]

        for b_idx, c_idx in enumerate(self.b_to_c):
            dst_params[c_idx] = params_b[b_idx]
            if src_params[c_idx] is None:
                src_params[c_idx] = params_b[b_idx]

        return jnp.array(src_params), jnp.array(dst_params)


    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        q_params_a = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_a)
        q_params_b = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_b) # HARD TYPO
        lj_params_a = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_a)
        lj_params_b = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_b)

        qlj_params_a = jnp.concatenate([
            jnp.reshape(q_params_a, (-1, 1)),
            jnp.reshape(lj_params_a, (-1, 2))
        ], axis=1)
        qlj_params_b = jnp.concatenate([
            jnp.reshape(q_params_b, (-1, 1)),
            jnp.reshape(lj_params_b, (-1, 2))
        ], axis=1)

        qlj_params_src, qlj_params_dst = self.interpolate_params(qlj_params_a, qlj_params_b)
        qlj_params = jnp.concatenate([qlj_params_src, qlj_params_dst])

        exclusion_idxs_a, scale_factors_a = nonbonded.generate_exclusion_idxs(
            self.mol_a,
            scale12=_SCALE_12,
            scale13=_SCALE_13,
            scale14=_SCALE_14
        )

        exclusion_idxs_b, scale_factors_b = nonbonded.generate_exclusion_idxs(
            self.mol_b,
            scale12=_SCALE_12,
            scale13=_SCALE_13,
            scale14=_SCALE_14
        )

        scale_factors_a = np.stack([scale_factors_a, scale_factors_a], axis=1)
        scale_factors_b = np.stack([scale_factors_b, scale_factors_b], axis=1)

        combined_scale_factors, combined_exclusion_idxs = self._parameterize_partial(
            scale_factors_a,
            scale_factors_b,
            exclusion_idxs_a,
            exclusion_idxs_b
        )

        # scale_factors are interpolated, but we don't support interpolation of scale
        # factors. So we assert that they are in fact identical at the endpoints

        scale_src = combined_scale_factors[:len(combined_scale_factors)//2]
        scale_dst = combined_scale_factors[len(combined_scale_factors)//2:]

        np.testing.assert_array_equal(scale_src, scale_dst)

        combined_scale_factors = scale_src

        # (ytz): we don't need exclusions between R_A and R_B will never see each other
        # under this decoupling scheme. They will always be at cutoff apart from each other.

        # plane_idxs: RA = Core = 0, RB = -1
        # offset_idxs: Core = 0, RA = RB = +1 
        combined_lambda_plane_idxs = np.zeros(self.get_num_atoms(), dtype=np.int32)
        combined_lambda_offset_idxs = np.zeros(self.get_num_atoms(), dtype=np.int32)

        for atom, group in enumerate(self.c_flags):
            if group == 0:
                # core atom
                combined_lambda_plane_idxs[atom] = 0
                combined_lambda_offset_idxs[atom] = 0
            elif group == 1:
                combined_lambda_plane_idxs[atom] = 0
                combined_lambda_offset_idxs[atom] = 1
            elif group == 2:
                combined_lambda_plane_idxs[atom] = -1
                combined_lambda_offset_idxs[atom] = 1
            else:
                assert 0

        beta = _BETA
        cutoff = _CUTOFF # solve for this analytically later

        nb = potentials.Nonbonded(
            combined_exclusion_idxs,
            combined_scale_factors,
            combined_lambda_plane_idxs,
            combined_lambda_offset_idxs,
            beta,
            cutoff
        ) 

        return qlj_params, potentials.InterpolatedPotential(nb, self.get_num_atoms(), qlj_params.size)

    @staticmethod
    def _assert_unique_idxs(idxs):
        seen = set()
        for atoms in idxs:
            if atoms[0] > atoms[-1]:
                atoms = atoms[::-1]
            atoms = tuple(atoms)
            assert atoms not in seen
            seen.add(atoms)

    def _parameterize_partial(self,
        params_a,
        params_b,
        idxs_a,
        idxs_b):


        offset = self.mol_a.GetNumAtoms()
        bonds_and_params_c = {}

        self._assert_unique_idxs(idxs_a)
        self._assert_unique_idxs(idxs_b)

        # (ytz): rule for how we combine bonded term is as follows
        # if a bonded term in A is comprised of only core atoms, then
        #   it will either be interpolated to zero or to that of the matching term in B (if it exists)
        # otherwise
        #   it will be kept at full strength and not decoupled
        for atoms_a, params_a in zip(idxs_a, params_a):
            if atoms_a[0] > atoms_a[-1]:
                atoms_a = atoms_a[::-1]
            key = tuple(atoms_a)
            if np.all(self.c_flags[atoms_a] == 0):

                # this relies on the semantic that the leading element is the force constant
                # for proper torsions every term need to goto zero.
                # tbd we can change this to ops.index[..., 0] so that the "leading zero"
                # gets set to zero.
                # [k, p, p] => [0, p, p] everything but proper torsions
                # [[k, p, p], [k, p, p], [k, p, p]] => [[0, p, p], [0, p, p],[0, p, p]] # proper torsions

                other = jax.ops.index_update(params_a, jax.ops.index[..., 0], 0)
            else:
                other = params_a

            bonds_and_params_c[key] = [params_a, other]

        for atoms_b, params_b in zip(idxs_b, params_b):
            # transform b_idxs into c_idxs
            atoms_b = [self.b_to_c[b] for b in atoms_b]
            if atoms_b[0] > atoms_b[-1]:
                atoms_b = atoms_b[::-1]
            key = tuple(atoms_b)

            if np.all(self.c_flags[atoms_b] == 0):
                other = jax.ops.index_update(params_b, jax.ops.index[..., 0], 0)
            else:
                other = params_b

            if key in bonds_and_params_c:
                bonds_and_params_c[key][1] = params_b
            else:
                bonds_and_params_c[key] = [other, params_b]

        idxs_c = []
        params_c_src = []
        params_c_dst = []

        for k, v in bonds_and_params_c.items():
            idxs_c.append(k)
            params_c_src.append(v[0])
            params_c_dst.append(v[1])

        params_c = jnp.concatenate([jnp.array(params_c_src), jnp.array(params_c_dst)])

        return params_c, np.array(idxs_c, dtype=np.int32)

    def parameterize_harmonic_bond(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.hb_handle, potentials.HarmonicBond)

    def parameterize_harmonic_angle(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.ha_handle, potentials.HarmonicAngle)

    @staticmethod
    def _flatten_torsions(torsion_params):
        """
        Slot a set of torsion params into a flat array.

        Parameters
        ----------
        torsion_params: np.ndarray Tx3
            Array of torsions with (k,phi,period)

        """
        # 6 possible periods, 4 possible phases
        params = np.zeros((_MAX_PERIOD, _MAX_PHASE), dtype=np.float64)

        for k, phase, period in torsion_params:
            # (ytz): this removes differentiability of the phase and period.
            phase_inc = jnp.pi/2
            phase_slot = jnp.lax.round(phase/phase_inc)
            assert jnp.abs(phase/phase_inc - phase_slot) < 1e-12

            # (ytz): note the extra minus one offset
            # periods are in the inclusive [1,6] range
            period_slot = jnp.lax.round(period) - 1
            assert jnp.abs((period - 1) - period_slot) < 1e-12

            params[jnp.int64(period_slot), jnp.int64(phase_slot)] = k

        return params.reshape(24)

    @staticmethod
    def _split_torsions(params):
        """
        Opposite of flatten torsions. Takes a flattened array and restores them.
        """

        params = np.reshape(params, (_MAX_PERIOD, _MAX_PHASE))
        dup_params = []

        for period_slot in range(_MAX_PERIOD):
            period = jnp.float64(period_slot + 1)
            for phase_slot in range(_MAX_PHASE):
                phase = phase_slot*(jnp.pi/2)
                k = params[period_slot, phase_slot]
                dup_params.append([k, phase, period])

        return dup_params

    @staticmethod
    def _deduplicate_torsions(idxs, params):
        idx_to_params = dict()

        for atoms, p in zip(idxs, params):
            atoms = tuple(atoms)
            if atoms not in dict():
                idx_to_params[atoms] = [p]
            else:
                idx_to_params[atoms].append(p)

        unique_idxs = []
        unique_params = []

        for atoms, p in idx_to_params.items():
            unique_idxs.append(atoms)
            unique_params.append(SingleTopology._flatten_torsions(p))

        return jnp.array(unique_idxs, dtype=jnp.int32), jnp.array(unique_params)

    @staticmethod
    def _duplicate_torsions(idxs, params):

        repeated_idxs = []
        repeated_params = []

        for t, p in zip(idxs, params):
            for pp in SingleTopology._split_torsions(p):
                repeated_idxs.append(t)
                repeated_params.append(pp)

        return jnp.array(repeated_idxs, dtype=np.int32), jnp.array(repeated_params)

    def parameterize_proper_torsion(self, ff_params):
        # (ytz): torsion need special handling as they contain duplicated idxs.
        # per discussion with jfass it was determined that the best, first-order
        # course of action is to only allow interpolation of parameters. This is
        # because for most of the openforcefield proper torsion parameters, periods
        # span from the integer range [1,6] (inclusive), and the phases span from
        # [0, 2pi) in increments of pi/2

        params_a, idxs_a = self.ff.pt_handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = self.ff.pt_handle.partial_parameterize(ff_params, self.mol_b)

        idxs_a, params_a = self._deduplicate_torsions(idxs_a, params_a)
        idxs_b, params_b = self._deduplicate_torsions(idxs_b, params_b)

        params_c, idxs_c = self._parameterize_partial(
            params_a,
            params_b,
            idxs_a,
            idxs_b
        )

        print("before dup", len(params_c)) # 20


        idxs_c, params_c = self._duplicate_torsions(idxs_c, params_c)

        # for 

        print("???", len(params_c))

        params_src = params_c[:len(params_c)//2]
        params_dst = params_c[len(params_c)//2:]

        # strip interpolations that go from 0 to 0
        final_idxs = []
        final_params_src = []
        final_params_dst = []
        for atoms, p_src, p_dst in zip(idxs_c, params_src, params_dst):
            if p_src[0] != 0 and p_dst[0] != 0:
                final_idxs.append(atoms)
                final_params_src.append(p_src)
                final_params_dst.append(p_dst)

        final_params_src = jnp.array(final_params_src)
        final_params_dst = jnp.array(final_params_dst)

        print(len(final_params_src), len(final_params_dst))

        final_params = jnp.concatenate([final_params_src, final_params_dst])

        print(len(final_params))

        return final_params, potentials.InterpolatedPotential(potentials.PeriodicTorsion(final_idxs), self.get_num_atoms(), final_params.size)

    def parameterize_improper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.it_handle, potentials.PeriodicTorsion)

    def _parameterize_bonded_term(self, ff_params, handle, potential):

        params_a, idxs_a = handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = handle.partial_parameterize(ff_params, self.mol_b)

        params_c, idxs_c = self._parameterize_partial(
            params_a,
            params_b,
            idxs_a,
            idxs_b
        )

        return params_c, potentials.InterpolatedPotential(potential(idxs_c), self.get_num_atoms(), params_c.size)