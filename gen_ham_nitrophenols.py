# standard lib imports
import argparse
import itertools
from multiprocessing import set_start_method, Pool
import warnings
from time import perf_counter

# 3rd party lib imports
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

# my lib import
import spectra_code.spectroscopic_maps as maps
from gen_ham import switching_function, _minimum_image

# Constants
a0 = 0.5291772
box_conversion = np.array(([a0] * 3) + ([1.] * 3))
hartree2wavenum = 219474.361363

class Universe:
    """ A class for handling the calculation of the vibrational Hamiltonian of a
        atomistic water simulation
        
    
    Parameters
    -----------
    args : argparse.Namespace
        See main() for details
    
    Attributes
    ----------
    universe : MDAnalysis.Universe
        Universe object containing MD trajectory
    
    waters : MDAnalysis.AtomGroup
        AtomGroup containing the water molecules in self.universe
    
    alcohols : MDAnalysis.AtomGroup
        AtomGroup containing non-water hydroxyl groups
    
    hydrogens : MDAnalysis.AtomGroup
        AtomGroup containing all OH-stretch hydrogens
        
    oxygens : MDAnalysis.AtomGroup
        AtomGroup containing all OH-stretch oxygens
    
    start_frame : int
        First frame of simulation to use
        
    end_frame : int
        Last frame of simulation to use
    
    nwaters : int
        Number of water molecules molecules in trajectory
    
    nstretch : int
        Number of stretches in trajectory
    
    nosc : int
        Number of oscillators in trajectory
        (alias for self.nstretch)
    
    natoms : int
        Number of atoms in trajectory

    cutoff : float
        Electric field cutoff radius in Å

    map_obj : spectra_code.spectroscopic_maps.Spectroscopic_Map
        Object for handling the choice of spectroscopic map for water
    
    al_map : spectra_code.spectroscopic_maps.Spectroscopic_Map
        Object for handling the choice of spectroscopic map for non-water hydroxyl groups
    
    atnums : np.array
        Atom index of atoms in universe
    
    charges : np.array
        Charges of atoms in universe
        
    res_len : int
        Number of atoms in each water (accounts for >3-site water models)
    
    inter_ndx : np.array
        Array of interatomic coupling indices
    
    intra_ndx: np.array
        Array of intramolecular coupling indices
    
    dist_mask : np.array
        Array of indices to mask when calculating electric fields
    
    exclude_ndx : np.array
        Array of indices to exclude when calculating electric fields
        (in addition to masking done by self.dist_mask)
    
    output_files : dict{str:str}
        Dictionary of output file names
    
    calc_types : list[str]
        List of outputs to calculate and save {'ham', 'dip', 'ram', 'sfg'}
    
    dists : np.array
        Distance matrix for the atoms in simulation
        
    interface_axis : {0, 1, 2}
        For SFG calculation dipoles, the axis along which to apply the switching function
    
    periodic : bool
        If True, assume the water slab falls across the pbc when calculating SFG switching function
        If False, assume the water slab does not fall across the pbc when calculating SFG switching function
        
    switching_cutoff : float
        r_c value passed to SFG switching function
    """
    def __init__(self, args):
        
        try:
            self.universe = mda.Universe(args.topology_file, args.trajectory_file)
        except ValueError:
            self.universe = mda.Universe(args.topology_file, args.trajectory_file, format='LAMMPSDUMP')

        self.correct_atom_types()
        
        self.start_frame = args.start_frame
        self.end_frame = args.end_frame
        if self.end_frame is None:
            self.end_frame = len(self.universe.trajectory)
        
        water_sel = args.water_selection
        if isinstance(water_sel, list):
            water_sel = ' '.join(water_sel)
        self.waters = self.universe.select_atoms(water_sel)
        
        al_hyd_str = args.alcohol_hydrogens
        if isinstance(al_hyd_str, list):
            al_hyd_str = ' '.join(al_hyd_str)
        al_hyd = self.universe.select_atoms(al_hyd_str)
        
        al_oxy_str = args.alcohol_oxygens
        if isinstance(al_oxy_str, list):
            al_oxy_str = ' '.join(al_oxy_str)
        al_oxy = self.universe.select_atoms(al_oxy_str)
        
        if len(al_hyd) != len(al_oxy):
            raise ValueError(f'Alcohol hydrogen and oxygen selections should be the same length. {len(al_hyd)} hydrogens and {len(al_oxy)} oxygens were selected using the strings "{al_hyd_str}" and "{al_oxy_str}", respectively.')
        
        self.alcohols = al_hyd + al_oxy
        
        self.hydrogens = self.waters.select_atoms('type H') + al_hyd
        self.oxygens = self.waters.select_atoms('type O') + al_oxy
        
        self.nwaters = np.int32(len(self.waters.residues))
        self.nstretch = (self.nwaters * 2) + len(al_hyd)
        self.nosc = self.nstretch
        self.natoms = len(self.universe.atoms)

        if args.water_model == 'TIP3P':
            self.cutoff = 8.31
        else:
            self.cutoff = 7.831
        if args.water_model == 'TIP3P':
            self.map_obj = maps.TIP3P_Map()
        elif args.water_model == 'SPCE':
            self.map_obj = maps.SPCE_Map()
        else:
            self.map_obj = maps.TIP4P_Map()
        
        if args.alcohol_map.lower() == 'onp':
            self.al_map = maps.oNP_Map(args.alcohol_map_ground_truth)
        elif args.alcohol_map.lower() == 'pnp':
            self.al_map = maps.pNP_Map(args.alcohol_map_ground_truth)
        
        self.atnums = self.universe.atoms.indices

        if not hasattr(self.universe.atoms, 'charges'):
            if args.charges:
                self.universe.add_TopologyAttr('charges')
                for itp in args.charges:
                    self.add_charges_from_itp(itp)
            else:
                raise Exception('No charge charge information available. Use -c or --charges to add charge information.')
        self.charges = self.universe.atoms.charges

        self.res_len = len(self.waters.residues[0].atoms)
        
        self.inter_ndx = np.meshgrid(np.arange(self.nstretch), np.arange(self.nstretch), indexing='ij')
        self.intra_ndx = [np.arange(0, self.nwaters*2, 2), np.arange(1, self.nwaters*2, 2)]
        
        self.dist_mask_water = (np.arange(self.nwaters*2)[..., None], np.full((2, self.nwaters, self.res_len), self.waters.residues.indices).reshape(self.nwaters*2, self.res_len, order='F'))
        self.dist_mask_al = (np.arange(self.nwaters*2, self.nstretch)[..., None], self.alcohols.indices.reshape(len(self.alcohols)//2, 2))
        
        exclude_sel = args.exclude_selection
        if isinstance(exclude_sel, list):
            exclude_sel = ' '.join(exclude_sel)
        self.exclude_ndx = self.universe.select_atoms(exclude_sel).indices

        self.output_files = {'ham':args.ham_file, 'dip':args.dip_file, 'sfg':args.sfg_file, 'ram':args.ram_file}
        all_keys = ['ham', 'dip', 'sfg', 'ram']
        for key in all_keys:
            if not self.output_files[key]:
                del self.output_files[key]
        self.calc_types = list(self.output_files.keys())
        self.dists = np.zeros((self.nstretch, self.natoms))

        print('Universe loaded!')
        
        self.interface_axis = ['x', 'y', 'z'].index(args.interface_axis.lower())
        self.periodic = args.periodic
        self.switching_cutoff = args.switching_cutoff
    
    def correct_atom_types(self):
        """ Changes the atom types of universe from OPLS names to element names """
        if self.universe.filename[-3:] == 'tpr':
            for atom in self.universe.atoms:
                if atom.name == 'OW':
                    atom.type = 'O'
                elif 'HW' in atom.name:
                    atom.type = 'H'
                else:
                    atom.type = 'MW'
        elif self.universe.filename[-4:] == 'data':
            for atom in self.universe.atoms:
                if atom.type == '1':
                    atom.type = 'O'
                elif atom.type == '2':
                    atom.type = 'H'
        elif 'OT' in self.universe.atoms.types:
            for atom in self.universe.atoms:
                if atom.type == 'OT':
                    atom.type = 'O'
                elif atom.type == 'HT':
                    atom.type = 'H'
        
    def add_charges_from_itp(self, itp_file):
        """ Reads in a .itp file to add atomic partial charges to self.Universe
        
        Parameters
        ----------
        itp_file : str
            Path to the desired .itp file
        """
        mol = mda.Universe(itp_file)
        resname = mol.residues[0].resname
        sel = self.universe.select_atoms(f'resname {resname}').residues
        for res in sel:
            res.atoms.charges = mol.atoms.charges


    
    def run_calc(self, pool, block_size=None):
        """ Runs calculation of Hamiltonian, dipoles, and polarizabilities
        
            Parameters
            ----------
            pool : multiprocessing.Pool
                Pool for running the calculations in parallel
            
            block_size : int
                The number of frames to calculate between writing to
                output files and clearing the Tensorflow session
        """
        if block_size is None:
            block_size = pool._processes
        
        frame_blocks = []
        for block_start in range(self.start_frame, self.end_frame, block_size):
            frame_blocks.append(list(range(block_start, block_start + block_size)))
        frame_blocks[-1] = list(range(frame_blocks[-1][0], self.end_frame))
        
        files = {key:open(self.output_files[key], 'wb') for key in self.calc_types}
        
        i = 0
        for block in frame_blocks:
            print('Running frames {}-{}'.format(block[0], block[-1]), flush=True)    
            pool_map = [[self, frame] for frame in block]
            results = pool.starmap(calc_ham_dip_ram, pool_map)
            self.write_results(results, files, block)
            i += 1
        
        for f in files.values():
            f.close()

        return None

    def write_results(self, results, files, frames):
        """ Writes results to output files
        
        Parameters
        ----------
        results : list
            List of tuples where each tuple consists of Hamiltonian,
            dipoles, and polarizabilties for a single frame
        files : list
            List of open files to write to
        frames : list
            List of frame indices corresponing to the results
        """
        
        for t in range(len(frames)):
            
            ham, dip, sfg, ram = results[t]
            frame_arr = np.array([frames[t]], dtype='float32')
            
            if 'ham' in self.calc_types:
                ham_flat = np.empty((int(self.nosc * (self.nosc + 1) / 2)), dtype='float32')
                temp = 0
                for i in range(self.nosc):
                    for j in range(i, self.nosc):
                        ham_flat[temp] = ham[i, j]
                        temp += 1
                frame_arr.tofile(files['ham'])
                ham_flat.tofile(files['ham'])
            
            if 'dip' in self.calc_types:
                dip_flat = np.empty((self.nosc * 3), dtype='float32')
                for i in range(3):
                    for j in range(self.nosc):
                        dip_flat[i * self.nosc + j] = dip[j,i]
                frame_arr.tofile(files['dip'])
                dip_flat.tofile(files['dip'])
            
            if 'sfg' in self.calc_types:
                sfg_flat = np.empty((self.nosc * 3), dtype='float32')
                for i in range(3):
                    for j in range(self.nosc):
                        sfg_flat[i * self.nosc + j] = sfg[j,i]
                frame_arr.tofile(files['sfg'])
                sfg_flat.tofile(files['sfg'])
            
            if 'ram' in self.calc_types:
                ram_flat = np.empty((self.nosc * 6), dtype='float32')
                for i in range(6):
                    for j in range(self.nosc):
                        ram_flat[i * self.nosc + j] = ram[j, i]
                frame_arr.tofile(files['ram'])
                ram_flat.tofile(files['ram'])
        
        return None


def calc_ham_dip_ram(universe, frame):
    """ Calculates the Hamiltonian, dipoles, and polarizabilities
        for a frame
        
        Parameters
        ----------
        universe : Universe
            Universe object, see above
            
        frame : int
            Frame of trajectory
        
        Returns
        -------
        hamiltonian : np.array
            Array of vibrational Hamiltonian
        
        dipole : np.array
            Array of transition dipoles
        
        sfg_dipole : np.array
            Array of transition dipoles with SFG switching function applied
        
        raman : np.array
            Array of transition polarizabilities
    """
    universe.universe.trajectory[frame]
    box = universe.universe.dimensions
    
    E_and_bonds = calc_all_E(universe)
    E = E_and_bonds[:, 0]
    bonds = E_and_bonds[:, 1:]
    del E_and_bonds
    
    w = np.concatenate((universe.map_obj.w_map(E[:universe.nwaters*2]), universe.al_map.w_map(E[universe.nwaters*2:])))
    mu = np.concatenate((universe.map_obj.mu_map(E[:universe.nwaters*2]), universe.al_map.mu_map(E[universe.nwaters*2:])))
       
    x = np.concatenate((universe.map_obj.x_map(w[:universe.nwaters*2]), universe.al_map.x_map(E[universe.nwaters*2:])))
    p = np.concatenate((universe.map_obj.p_map(w[:universe.nwaters*2]), universe.al_map.p_map(E[universe.nwaters*2:])))
    
    dipole = np.zeros((universe.nosc, 3), dtype=np.float32)
    dipole[:universe.nstretch] = mu[..., None] * x[..., None] * bonds
    
    if 'sfg' in universe.calc_types:
        if universe.interface_axis == 0:
            R = np.array([[0,0,-1],[0,1,0],[-1,0,0]])
            bonds = np.einsum('ij,...j', R, bonds)
        elif universe.interface_axis == 1:
            R = np.array([[1,0,0],[0,0,1],[0,-1,0]])
            bonds = np.einsum('ij,...j', R, bonds)
        z = universe.oxygens.positions[:, universe.interface_axis]
        if universe.periodic:
            f_z = switching_function(z, box=universe.universe.dimensions[universe.interface_axis], r_c=universe.switching_cutoff)
        else:
            f_z = switching_function(z, r_c=universe.switching_cutoff)
        f_z = np.concatenate((np.full((2, universe.nwaters), f_z[:universe.nwaters]).reshape(universe.nwaters*2, order='F'), f_z[universe.nwaters:]))
        sfg_dipole = dipole * f_z[..., None]
    else:
        sfg_dipole = None
    
    if 'ram' in universe.calc_types:
        raman = np.zeros((universe.nosc, 6), dtype=np.float32)
        i = 0
        for j in range(3):
            for k in range(j, 3):
                if j == k:
                    raman[:universe.nstretch, i] = 4.6 * x * (bonds[:, j] ** 2) + x
                else:
                    raman[:universe.nstretch, i] = 4.6 * x * bonds[:, j] * bonds[:, k]
                i += 1
    else:
        raman = None
    
    if 'ham' in universe.calc_types:
        hamiltonian = np.zeros((universe.nstretch, universe.nstretch))
        
        oxy_pos = np.concatenate([np.full((2, universe.nwaters, 3), universe.oxygens[:universe.nwaters].positions).reshape((universe.nwaters*2,3), order='F'), universe.oxygens[universe.nwaters:].positions])
        d = oxy_pos + (0.67 * bonds)
        dists = distance_array(d, d, box=box)
        
        i, j = universe.inter_ndx
        n_hat = _minimum_image(d[i] - d[j], box) / dists[i, j].reshape(universe.nstretch, universe.nstretch, 1)
        dists /= a0
        k_inter = (mu[i] * mu[j] * (np.einsum('...i, ...i', bonds[i], bonds[j]) - (3 * np.einsum('...i, ...i', bonds[i], n_hat) * np.einsum('...i, ...i', bonds[j], n_hat)))) / dists[i, j] ** 3
        inter = k_inter * x[i] * x[j] * hartree2wavenum
        hamiltonian[i,j]  = inter
        
        np.fill_diagonal(hamiltonian, w)
        
        i, j = universe.intra_ndx
        intra = universe.map_obj.intra_map(E[i], E[j], x[i], x[j], p[i], p[j])
        hamiltonian[i,j] = intra
        hamiltonian[j,i] = intra
        
    else:
        hamiltonian = None
    
    return hamiltonian, dipole, sfg_dipole, raman
    
def calc_all_E(universe):
    """ Calculates electric field projections and OH stretch unit vectors
        for all stretches in the set frame
    
    Parameters
    ----------
    universe : Universe
        Universe object, see above
    
    Returns
    -------
    return_arr : np.array
        Array containing electric field projections and OH unit vectors
    """
    
    E_vector = electric_field_vectors(universe)
                
    oxy_pos = np.concatenate([np.full((2, universe.nwaters, 3), universe.oxygens[:universe.nwaters].positions).reshape((universe.nwaters*2,3), order='F'), universe.oxygens[universe.nwaters:].positions])
    bonds = _minimum_image(universe.hydrogens.positions - oxy_pos, universe.universe.dimensions)
    bond_hats = bonds / np.linalg.norm(bonds, axis=1)[..., None]
    E = np.einsum('...i, ...i', bond_hats, E_vector)
    
    return_arr = np.zeros((universe.nstretch,4))
    return_arr[:, 0] = E
    return_arr[:, 1:] = bond_hats
    
    return return_arr

def electric_field_vectors(universe):
    """ Calculates electric field projections and OH stretch unit vectors
        for all stretches in the set frame
    
    Parameters
    ----------
    universe : Universe
        Universe object, see above
    
    Returns
    -------
    E : np.array
        Electric field vectors
    """
    
    h_pos = universe.hydrogens.positions
    atom_pos = universe.universe.atoms.positions
    box = universe.universe.dimensions
    
    box /= box_conversion
    h_pos /= a0
    atom_pos /= a0
    
    dists = distance_array(h_pos, atom_pos, box=box)
    universe.dists = np.copy(dists)
    dists[universe.dist_mask_water] = np.nan
    dists[universe.dist_mask_al] = np.nan
    dists[:,universe.exclude_ndx] = np.nan
    for i in range(universe.nstretch):
        within_cutoff = (dists[i] < (universe.cutoff / a0)).nonzero()
        j = np.reshape(universe.universe.atoms[within_cutoff].residues.atoms.indices, -1)
        j = np.setdiff1d(universe.atnums, j)
        dists[i,j] = np.nan
 
    q = universe.charges
    
    r = _minimum_image(h_pos[:, None, :] - np.full((universe.nstretch, universe.natoms, 3), atom_pos), box)
    
    with warnings.catch_warnings(action='ignore', category=RuntimeWarning):
        E = np.nansum(q[None, ..., None] * (r / (dists**3)[..., None]), axis=1)
    
    return E


def main():
    # Default on macOS and Windows but not on Linux
    set_start_method('spawn')

    parser = argparse.ArgumentParser()
    
    file_group = parser.add_argument_group('Input and output files')
    file_group.add_argument('-top', '--topology_file', required=True, metavar='FILENAME', help='Topology file for MD simulation containing water')
    file_group.add_argument('-trj', '--trajectory_file', metavar='FILENAME', help='Trajectory file for MD simulation containing water')
    file_group.add_argument('-f', '--ham_file', default='hamil.bin', metavar='FILENAME', help='The output Hamiltonian trajectory', type=lambda x : None if x == 'None' else x)
    file_group.add_argument('-d', '--dip_file', default='dipole.bin', metavar='FILENAME', help='The output transition dipole trajectory', type=lambda x : None if x == 'None' else x)
    file_group.add_argument('-r', '--ram_file', default='raman.bin', metavar='FILENAME', help='The output transition polarizability trajectory', type=lambda x : None if x == 'None' else x)
    file_group.add_argument('-sfg', '--sfg_file', default=None, metavar='FILENAME', help='Transition dipole trajecotory scaled for calculating SFG spectra', type=lambda x : None if x == 'None' else x)
    
    system_group = parser.add_argument_group('Settings for handling system specifics')
    system_group.add_argument('-s', '--start_frame', default=0, type=int, metavar='INT', help='The first frame of the trajectory to use')
    system_group.add_argument('-e', '--end_frame', default=None, type=int, metavar='INT', help='The final frame of the trajectory to use')
    system_group.add_argument('-w', '--water_model', default='TIP4P', metavar='{TIP4P, E3B2, TIP3P, SPCE}', help='The water model used to run the simulation')
    system_group.add_argument('-ws', '--water_selection', default='all', metavar='STR', type=str, help='MDAnalysis selection string for the water molecule atoms', nargs='+')
    system_group.add_argument('-a', '--alcohol_map', default=None, metavar='{onp, pnp}', help='The map to apply to alcohols')
    system_group.add_argument('-gt', '--alcohol_map_ground_truth', default='B3LYP', metavar='{B3LYP, AIMNet}')
    system_group.add_argument('-a_oxy', '--alcohol_oxygens', required=True, metavar='STR', type=str, help='MDAnalysis selection string for the alcohol oxygens', nargs='+')
    system_group.add_argument('-a_hyd', '--alcohol_hydrogens', required=True, metavar='STR', type=str, help='MDAnalysis selection string for the alcohol hydrogens', nargs='+')
    system_group.add_argument('-i', '--interface_axis', default='z', metavar='{x, y, z}', help='Axis perpendicular to interface for interfacial simulations')
    system_group.add_argument('-c', '--charges', default=None, metavar='ITP_FILE', nargs='*',  help='.json file containing partial charges for each molecule')
    system_group.add_argument('-p', '--periodic', action='store_true', help='Water slab falls across periodic boundary')
    system_group.add_argument('-es', '--exclude_selection', default=None, metavar='STR', type=str, help='MDAnalysis selection string to ignore when calculating electric field', nargs='+')
    
    parallel_group = parser.add_argument_group('Parallelization settings')
    parallel_group.add_argument('-n', '--n_procs', default=8, type=int, metavar='INT', help='The number of parallel processes to run')
    parallel_group.add_argument('-b', '--block_size', default=None, type=int, metavar='INT', help='The number of frames to calculate between writing to output files; smaller values will decrease memory profile; should be ≥ n_procs')
    
    parser.add_argument('-rc', '--switching_cutoff', type=float, metavar='FLOAT', help='Switching function R_c value in Å', default=4.0)

    args = parser.parse_args()
    
    universe = Universe(args)
    print('Created Universe')
    
    print('Creating pool') 
    pool = Pool(args.n_procs)

    print('Pool created, starting calculation', flush=True)

    universe.run_calc(pool, args.block_size)
    
    print('Job complete!')


if __name__ == '__main__':
    main()
