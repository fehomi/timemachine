import numpy as np
from simtk import unit
from simtk.openmm import app, Vec3


def strip_units(coords):
    return unit.Quantity(np.array(coords / coords.unit), coords.unit)

def build_protein_system(host_pdbfile):

    # host_pdbfile = general_cfg['protein_pdb']
    host_ff = app.ForceField('amber99sbildn.xml', 'tip3p.xml')
    host_pdb = app.PDBFile(host_pdbfile)

    modeller = app.Modeller(host_pdb.topology, host_pdb.positions)
    host_coords = strip_units(host_pdb.positions)

    padding = 1.0
    box_lengths = np.amax(host_coords, axis=0) - np.amin(host_coords, axis=0)
    box_lengths = box_lengths.value_in_unit_system(unit.md_unit_system)
    box_lengths = box_lengths+padding
    box = np.eye(3, dtype=np.float64)*box_lengths

    modeller.addSolvent(host_ff, boxSize=np.diag(box)*unit.nanometers, neutralize=False)
    solvated_host_coords = strip_units(modeller.positions)

    nha = host_coords.shape[0]
    nwa = solvated_host_coords.shape[0] - nha

    print(nha, "protein atoms", nwa, "water atoms")
    solvated_host_system = host_ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )

    return solvated_host_system, solvated_host_coords, nwa, nha, box


def build_water_system(box_width):
    # if model not in supported_models:
        # raise Exception("Specified water model '%s' is not in list of supported models: %s" % (model, str(supported_models)))

    # Load forcefield for solvent model and ions.
    # force_fields = ['tip3p.xml']
    # if ionic_strength != 0.0*unit.molar:
        # force_fields.append('amber99sb.xml')  # For the ions.
    ff = app.ForceField('tip3p.xml')

    # Create empty topology and coordinates.
    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)

    # Create new Modeller instance.
    m = app.Modeller(top, pos)

    boxSize = Vec3(box_width, box_width, box_width)*unit.nanometers
    # boxSize = unit.Quantity(numpy.ones([3]) * box_edge / box_edge.unit, box_edge.unit)
    m.addSolvent(ff, boxSize=boxSize, model='tip3p')

    system = ff.createSystem(
        m.getTopology(),
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )

    positions = m.getPositions()

    positions = unit.Quantity(np.array(positions / positions.unit), positions.unit)

    # pdb_str = io.StringIO()
    # fname = "debug_water.pdb"
    # fhandle = open(fname, "w")
    # PDBFile.writeHeader(m.getTopology(), fhandle)
    # PDBFile.writeModel(m.getTopology(), positions, fhandle, 0)
    # PDBFile.writeFooter(m.getTopology(), fhandle)

    return system, positions, np.eye(3)*box_width

    # assert 0