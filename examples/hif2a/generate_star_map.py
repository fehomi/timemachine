# Construct a star map for the fep-benchmark hif2a ligands
import sys
from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path
from rdkit import Chem

from pickle import dump

import timemachine
from timemachine.parser import TimemachineConfig

from fe import topology
from fe.utils import convert_uIC50_to_kJ_per_mole
from fe.free_energy import RelativeFreeEnergy
from fe.topology import AtomMappingError

# 0. Get force field
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from rdkit.Chem import rdFMCS

import numpy as np
from fe.atom_mapping import (
    get_core_by_geometry,
    get_core_by_matching,
    get_core_by_mcs,
    get_core_by_smarts,
    mcs_map
)

# Get the root off of the timemachine install
root = Path(timemachine.__file__).absolute().parent


def get_mol_id(mol):
    return mol.GetProp('ID')


def _compute_label(mol_a, mol_b):
    """ Compute labeled ddg (in kJ/mol) from the experimental IC50 s """

    prop_name = "IC50[uM](SPA)"
    try:
        label_dG_a = convert_uIC50_to_kJ_per_mole(float(mol_a.GetProp(prop_name)))
        label_dG_b = convert_uIC50_to_kJ_per_mole(float(mol_b.GetProp(prop_name)))
    except KeyError as e:
        raise RuntimeError(f"Couldn't access IC50 label for either mol A or mol B, looking at {prop_name}")

    label = label_dG_b - label_dG_a

    return label


# filter by transformation size
def transformation_size(rfe: RelativeFreeEnergy):
    n_A, n_B, n_MCS = rfe.mol_a.GetNumAtoms(), rfe.mol_b.GetNumAtoms(), len(rfe.core)
    return (n_A + n_B) - 2 * n_MCS


def get_core_by_permissive_mcs(mol_a, mol_b):
    mcs_result = rdFMCS.FindMCS(
        [mol_a, mol_b],
        timeout=30,
        threshold=1.0,
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        completeRingsOnly=True,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=True,
    )
    query = mcs_result.queryMol

    # fails distance assertions
    # return get_core_by_mcs(mol_a, mol_b, query)

    inds_a = mol_a.GetSubstructMatches(query)[0]
    inds_b = mol_b.GetSubstructMatches(query)[0]
    core = np.array([inds_a, inds_b]).T

    return core


def _get_match(mol, query):
    matches = mol.GetSubstructMatches(query)
    return matches[0]


def _get_core_by_smarts_wo_checking_uniqueness(mol_a, mol_b, core_smarts):
    """no atom mapping errors with this one, but the core size is smaller"""
    query = Chem.MolFromSmarts(core_smarts)

    return np.array([_get_match(mol_a, query), _get_match(mol_b, query)]).T

core_strategies = {
    'custom_mcs': lambda a, b : get_core_by_mcs(a, b, mcs_map(a, b).queryMol),
    'any_mcs': lambda a, b : get_core_by_permissive_mcs(a, b),
    'geometry': lambda a, b: get_core_by_geometry(a, b, threshold=0.5),
    'smarts_core_1': lambda a, b: get_core_by_smarts(a, b, core_smarts=bicyclic_smarts_pattern),
    'smarts_core_2': lambda a, b: _get_core_by_smarts_wo_checking_uniqueness(a, b, core_smarts=core_2_smarts)
}


def generate_star(hub, spokes, transformation_size_threshold=2, core_strategy='geometry'):
    transformations = []
    error_transformations = []
    for spoke in spokes:
        core = core_strategies[core_strategy](hub, spoke)

        # TODO: reduce overlap between get_core_by_smarts and get_core_by_mcs
        # TODO: replace big ol' list of get_core_by_*(mol_a, mol_b, **kwargs) functions with something... classy

        try:
            single_topology = topology.SingleTopology(hub, spoke, core, forcefield)
            rfe = RelativeFreeEnergy(single_topology, label=_compute_label(hub, spoke))
            transformations.append(rfe)
        except AtomMappingError as e:
            # note: some of transformations may fail the factorizability assertion here:
            # https://github.com/proteneer/timemachine/blob/2eb956f9f8ce62287cc531188d1d1481832c5e96/fe/topology.py#L381-L431
            print(f'atom mapping error in transformation {get_mol_id(hub)} -> {get_mol_id(spoke)}!')
            print(e)
            error_transformations.append((hub, spoke, core))

    print(f'total # of edges that encountered atom mapping errors: {len(error_transformations)}')

    # filter to keep just the edges with very small number of atoms changing
    easy_transformations = [rfe for rfe in transformations if transformation_size(rfe) <= transformation_size_threshold]

    return easy_transformations, error_transformations

def mol_matches_core(mol, core_query) -> bool:
    res = mol.GetSubstructMatches(core_query)
    if len(res) > 1:
        mol_id = get_mol_id(mol)
        print(f"Mol {mol_id} matched core multiple times")
    return len(res) == 1

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate FEP edge map")
    parser.add_argument("config", help="YAML configuration")
    args = parser.parse_args()
    config = TimemachineConfig.from_yaml(args.config)
    if config.map_generation is None:
        print("No map generation configuration provided")
        sys.exit(1)
    map_config = config.map_generation

    with open(map_config.forcefield) as f:
        ff_handlers = deserialize_handlers(f.read())
    forcefield = Forcefield(ff_handlers)

    mols = []
    for lig_path in map_config.ligands:
        supplier = Chem.SDMolSupplier(lig_path, removeHs=False)
        mols.extend(list(supplier))

    # In the future hopefully we can programmatically find the cores rather specifying
    cores = map_config.cores

    core_sets = defaultdict(list)
    for core in cores:
        core_query = Chem.MolFromSmarts(core)
        for i in range(len(mols)):
            mol = mols[i]
            if mol is None:
                continue
            if mol_matches_core(mol, core_query):
                core_sets[core].append(mol)
                mols[i] = None
    if any(mols):
        print("Not all mols matched the provided cores")
        leftover = [get_mol_id(mol) for mol in filter(None, mols)]
        print(f"Mols that didn't match cores: {leftover}")

    transformation_size_threshold = 3
    all_edges = []
    hubs_specified = map_config.hubs is not None and len(map_config.cores) == len(map_config.hubs)
    for i, core in enumerate(cores):
        mols = core_sets[core]
        if hubs_specified and map_config.hubs[i]:
            hub_idx = [get_mol_id(mol) for mol in mols].index(map_config.hubs[i])
            hub = mols[hub_idx]
            mols.pop(hub_idx)
        else:
            raise NotImplementedError("Requires a hub right now")
        edges, errors = generate_star(hub, mols, transformation_size_threshold=transformation_size_threshold, core_strategy=map_config.strategy)
        all_edges.extend(edges)
        with open(f"core_{i}_error_transformations.pkl", "wb") as f:
            dump(errors, f)

    # serialize
    with open(map_config.output, 'wb') as f:
        dump(all_edges, f)
