# Last updated: 12/17/24

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

# Warning filters
warnings.filterwarnings('ignore', '.*X does not have valid feature names, but StandardScaler was fitted with feature names.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, lineno=408)

# Settings that should stay the same unless you've intentionally changed things
CUTOFF = 7.831
input_size = 306
oxy_acsf_len = 53
hyd_acsf_len = 50
oxy_g2_params = [[0.001, 0.0], [0.01, 0.0], [0.03, 0.0], [0.06, 0.0], 
                 [0.15, 0.9], [0.15, 1.9], [0.3, 0.9], [0.3, 1.9],
                 [0.6, 0.9], [0.6, 1.9], [1.5, 0.9], [1.5, 1.9]]
oxy_g4_params = [[0.001, 4.0, -1.0], [0.001, 4.0, 1.0], [0.01, 4.0, -1.0],
                 [0.01, 4.0, 1.0], [0.03, 1.0, -1.0], [0.03, 1.0, 1.0],
                 [0.07, 1.0, -1.0], [0.07, 1.0, 1.0], [0.2, 1.0, 1.0]]
hyd_g2_params = [[0.001, 0.0], [0.01, 0.0], [0.03, 0.0], [0.06, 0.0], 
                 [0.15, 0.9], [0.15, 4.0], [0.3, 0.9], [0.3, 4.0],
                 [0.6, 0.9], [0.6, 4.0], [1.5, 0.9], [1.5, 4.0]] 
hyd_g4_params = [[0.001, 4.0, -1.0], [0.001, 4.0, 1.0], [0.01, 4.0, -1.0],
                 [0.01, 4.0, 1.0], [0.03, 1.0, -1.0], [0.03, 1.0, 1.0],
                 [0.07, 1.0, -1.0], [0.07, 1.0, 1.0]] 
acsf_cutoff = 12.


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
    universe : MDAnlysis.Universe
        Universe object containing atomistic water trajectory
    
    start_frame : int
        First frame of simulation to use
        
    end_frame : int
        Last frame of simulation to use
    
    atnums : np.array
        Atom index of atoms in universe
    
    types : np.array
        Atom types of atoms in universe
    
    charges : np.array
        Charges of atoms in universe
    
    nres : int
        Number of molecules in trajectory
    
    nstretch : int
        Number of stretches in trajectory
    
    natoms : int
        Number of atoms in trajectory
    
    res_len : int
        Number of atoms in each molecule
    
    add_M : bool
        If True, TIP4P charge position must be calculated
        If False, TIP4P charge positions are included in trajectory
    
    hydrogens : MDAnalysis.AtomGroup
        AtomGroup of the hydrogens in the trajectory
    
    oxygens : MDAnalysis.AtomGroup
        AtomGroup of the oxygens in the trajectory
    
    inter_ndx : np.array
        Array of interatomic coupling indices
    
    intra_ndx: np.array
        Array of intramolecular coupling indices
    
    dist_mask : np.array
        Array of indices to mask when calculating electric fields
    
    ham_file : str
        File name of the Hamiltonian output file
    
    dip_file : str
        File name of the transition dipole output file
    
    sfg_dip_file : str
        File name of the transition dipole output file with SFG switching function applied
    
    ram_file : str
        File name of the transition polarizability output file
    
    model : {None, predition_model}
        If using ∆-ML map, trained TensorFlow model
    
    dists : np.array
        Distance matrix for the atoms in simulation
    """
    def __init__(self, args):
        
        try:
            self.universe = mda.Universe(args.topology_file, args.trajectory_file)
        except ValueError:
            self.universe = mda.Universe(args.topology_file, args.trajectory_file, format='LAMMPSDUMP')

        self.correct_atom_types()
        
        self.start_frame = args.start_frame
        self.end_frame = args.end_frame
        
        self.waters = self.universe.select_atoms(f'resname {args.water_name}')
        
        self.hydrogens = self.waters.select_atoms('type H')
        self.oxygens = self.waters.select_atoms('type O')
        
        self.nres = np.int32(len(self.waters.residues))
        self.nstretch = self.nres * 2
        self.nosc = self.nstretch
        self.natoms = len(self.waters.atoms)

        if args.water_model == 'TIP3P':
             self.hydrogens.charges = np.full(self.nstretch, 0.52)
             self.oxygens.charges = np.full(self.nres, -1.04)
        
        self.atnums = self.universe.atoms.indices
        self.charges = self.waters.atoms.charges
        
        self.fermi = args.fermi
        
        if self.fermi:
            self.nosc += self.nres

        
        self.res_len = len(waters.residues[0].atoms)
        self.add_M = self.res_len == 3
        
        self.inter_ndx = np.meshgrid(np.arange(self.nstretch), np.arange(self.nstretch), indexing='ij')
        self.intra_ndx = [np.arange(0, self.nstretch, 2), np.arange(1, self.nstretch, 2)]
        if self.fermi:
            self.fermi_ndx = [np.full((2, self.nres), np.arange(self.nstretch, self.nosc)).flatten(order='F'), np.arange(self.nstretch)]
        
        self.dist_mask = (np.arange(self.nstretch)[..., None], np.full((2, self.nres, self.res_len), self.waters.residues.indices).reshape(self.nstretch, self.res_len, order='F'))
        

        self.output_files = {'ham':args.ham_file, 'dip':args.dip_file, 'sfg':args.sfg_file, 'ram':args.ram_file}
        for key in output_files.keys():
            if not output_files[key]:
                del output_files[key]
        self.calc_types = list(output_files.keys())
        
        self.model_file = args.model_file
        if self.model_file:
            self.setup_delML(args)
        self.dists = np.zeros((self.nstretch, self.natoms))

        print('Universe loaded!')
        print(self.model)
        
        self.interface_axis = ['x', 'y', 'z'].index(args.interface_axis.lower())
    
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


    def setup_delML(self, args):
        """ Sets up attributes needed for using ∆-ML map """
        
        import pickle
        
        from ase import Atoms
        from dscribe.descriptors import ACSF
        from tensorflow.keras.models import load_model
        from tensorflow.keras.backend import clear_session

        from sklearnex import patch_sklearn
        patch_sklearn()
        
        with open(args.x_scaler, 'rb') as file:
            self.x_scaler = pickle.load(file)
        with open(args.y_scaler, 'rb') as file:
            self.y_scaler = pickle.load(file)
        self.model = load_model(self.model_file)
        self.acsf_oxy = ACSF(species=['O', 'H'], r_cut=acsf_cutoff, g2_params=oxy_g2_params, g4_params=oxy_g4_params, periodic=True)
        self.acsf_hyd = ACSF(species=['O', 'H'], r_cut=acsf_cutoff, g2_params=hyd_g2_params, g4_params=hyd_g4_params, periodic=True)
        self.oxy_ndx = self.oxygens.indices 
        self.hyd_ndx = self.hydrogens.indices
        if self.add_M:
            self.ase = Atoms(symbols=self.waters.atoms.types, positions=self.waters.positions, cell=self.universe.dimensions[:3], pbc=True)
        else:
            self.OH_ndx = np.union1d(self.oxy_ndx, self.hyd_ndx)
            self.ase = Atoms(symbols=['O', 'H', 'H']*self.nres, positions=self.universe.atoms[self.OH_ndx].positions, cell=self.universe.dimensions[:3], pbc=True)
            ndx = np.arange(3*self.nres)
            self.oxy_ndx = ndx[::3]
            self.hyd_ndx = np.setdiff1d(ndx, self.oxy_ndx)
        self.oxy_mask = (np.arange(self.nstretch), np.arange(self.nstretch) // 2)
        self.hyd_mask = (np.arange(self.nstretch), np.arange(self.nstretch).reshape(self.nres, 2)[:, ::-1].flatten())


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
        
        files = {key:open(self.file_names[key], 'wb') for key in self.calc_types}
        
        i = 0
        for block in frame_blocks:
            print('Running frames {}-{}'.format(block[0], block[-1]), flush=True)    
            pool_map = [[self, frame] for frame in block]
            results = pool.starmap(calc_ham_dip_ram, pool_map)
            self.write_results(results, files, block)
            i += 1
            if self.model and i%5 == 0:
                clear_session() # Prevents memory leaks from Keras backend
                self.model = load_model(self.model_file)
        
        for f in files:
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
            
            if 'ham' in self.calc_types:
                frame_arr = np.array([frames[t]], dtype='float32')
                ham_flat = np.empty((int(self.nstretch * (self.nstretch + 1) / 2)), dtype='float32')
                temp = 0
                for i in range(self.nstretch):
                    for j in range(i, self.nstretch):
                        ham_flat[temp] = ham[i, j]
                        temp += 1
                frame_arr.tofile(files['ham'])
                ham_flat.tofile(files['ham'])
            
            if 'dip' in self.calc_types:
                dip_flat = np.empty((self.nstretch * 3), dtype='float32')
                for i in range(3):
                    for j in range(self.nstretch):
                        dip_flat[i * self.nstretch + j] = dip[j,i]
                frame_arr.tofile(files['dip'])
                dip_flat.tofile(files['dip'])
            
            if 'sfg' in self.calc_types:
                sfg_flat = np.empty((self.nstretch * 3), dtype='float32')
                for i in range(3):
                    for j in range(self.nstretch):
                        sfg_flat[i * self.nstretch + j] = sfg[j,i]
                frame_arr.tofile(files['sfg'])
                sfg_flat.tofile(files['sfg'])
            
            if 'ram' in self.calc_types:
                ram_flat = np.empty((self.nstretch * 6), dtype='float32')
                for i in range(6):
                    for j in range(self.nstretch):
                        ram_flat[i * self.nstretch + j] = ram[j, i]
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
        
        raman : np.array
            Array of transition polarizabilities
    """
    universe.universe.trajectory[frame]
    box = universe.universe.dimensions
    
    E_and_bonds = calc_all_E(universe)
    E = E_and_bonds[:, 0]
    E_bend = E_and_bonds[:, 1]
    bonds = E_and_bonds[:, 2:]
    del E_and_bonds
        
    w = 3760.2 - (3541.7 * E) - (152677 * (E ** 2))
    x = 0.19285 - (1.7261e-5 * w)
    p = 1.6466 + (5.7692e-4 * w)
    mu = 0.1646 + (11.39 * E) + (63.41 * (E ** 2))
    
    if universe.fermi:
        w_bend = 3132.78 + 6086.31 * E_bend

    if universe.model:
        w, mu = correct_map(universe, w, mu)
    
    dipole = mu[..., None] * x[..., None] * bonds
    
    if 'sfg' in universe.calc_types:
        z = universe.oxygens.positions[:, universe.interface_axis]
        slab_center = np.mean(z)
        f_z = switching_function(z, slab_center)
        f_z = np.full((2, len(f_z)), f_z).reshape(universe.nstretch, order='F')
        sfg_dipole = dipole * f_z
    else:
        sfg_dipole = None
    
    if 'ram' in universe.calc_types:
        raman = np.zeros((universe.nstretch, 6))
        i = 0
        for j in range(3):
            for k in range(j, 3):
                if j == k:
                    raman[:, i] = 4.6 * x * (bonds[:, j] ** 2) + x
                else:
                    raman[:, i] = 4.6 * x * bonds[:, j] * bonds[:, k]
                i += 1
    else:
        raman = None
    
    if 'ham' in universe.calc_types:
        hamiltonian = np.zeros((universe.nstretch, universe.nstretch))
        
        oxy_pos = np.full((2, universe.nres, 3), universe.oxygens.positions).reshape(universe.nstretch, 3, order='F')
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
        intra = ((-1361 + (27165 * (E[i] + E[j]))) * x[i] * x[j]) - (1.887 * p[i] * p[j])
        hamiltonian[i,j] = intra
        hamiltonian[j,i] = intra
        
        if universe.fermi:
            stretch_hamiltonian = hamiltonian
            i, j = np.indices(stretch_hamiltonian.shape)
            
            hamiltonian = np.zeros((universe.nosc, universe.nosc))
            hamiltonian[i, j] = stretch_hamiltonian

            i = np.arange(universe.nstretch, universe.nosc)
            hamiltonian[i, i] = w_bend

            i, j = self.fermi_ndx
            hamiltonian[i, j] = 25.
            hamiltonian[j, i] = 25.
        
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
    
    oxy_pos = np.full((2, universe.nres, 3), universe.oxygens.positions).reshape((universe.nstretch,3), order='F')
    bonds = _minimum_image(universe.hydrogens.positions - oxy_pos, universe.universe.dimensions)
    bond_hats = bonds / np.linalg.norm(bonds, axis=1)[..., None]
    E = np.einsum('...i, ...i', bond_hats, E_vector)
    
    return_arr = np.zeros((universe.nstretch,5))
    return_arr[:, 0] = E
    return_arr[:, 2:] = bond_hats
    
    if universe.fermi:
        bipoints = universe.oxygens.positions + (1.335 * bonds[::2]) + (1.335 * bonds[1::2])
        bipoints = np.full((2, universe.nres, 3), bipoints).reshape((universe.nstretch,3), order='F')
        vecs = _minimum_image(universe.hydrogens.positions - bipoints, universe.universe.dimensions)
        vec_hats = vecs / np.linalg.norm(vecs, axis=1)[..., None]
        E_vecs = np.einsum('...i, ...i', vec_hats, E_vector)
        E_bend = E_vecs.reshape(2, universe.nres, order='F').sum(axis=0)
        return_arr[:, 1] = E_bend
    
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
    
    if universe.add_M:
        oxy_pos = atom_pos[self.universe.oxygens.indices]
        OH1 = _minimum_image(h_pos[::2] - oxy_pos, box) * 0.128012065
        OH2 = _minimum_image(h_pos[1::2] - oxy_pos, box) * 0.128012065
        atom_pos[self.universe.oxygens.indices] = oxy_pos + OH1 + OH2
    
    box /= box_conversion
    h_pos /= a0
    atom_pos /= a0
    
    dists = distance_array(h_pos, atom_pos, box=box)
    universe.dists = np.copy(dists)
    dists[universe.dist_mask] = np.nan
    for i in range(universe.nstretch):
        within_cutoff = (dists[i] < (CUTOFF / a0)).nonzero()
        j = np.reshape(universe.universe.atoms[within_cutoff].residues.indices, -1)
        j = np.setdiff1d(universe.atnums, j)
        dists[i,j] = np.nan
 
    q = universe.charges
    
    r = _minimum_image(h_pos[:, None, :] - np.full((universe.nstretch, universe.natoms, 3), atom_pos), box)
    
    E = np.nansum(q[None, ..., None] * (r / (dists**3)[..., None]), axis=1)
    
    return E


def _minimum_image(position, box):
    """ Calculates minimum image across periodic boundary conditions
    
    Parameters
    ----------
    position : np.array
        n-by-3 array of positions or vectors
    box : np.array
        Box dimensions for calculation of periodic boundary conditions
    
    Returns
    -------
    position : np.array
        Original positions with periodic boundary counditions
        accounted for
    
    Notes
    -------
        Will fail if box length is set to 0
    """
    box = box[:3]
    s = position / box
    position = box * (s - np.round(s))
    return position

def correct_map(universe, w, mu):
    if universe.add_M:
        universe.ase.positions = universe.waters.positions / a0
        dists = universe.dists[:, ::3]
    else:
        universe.ase.positions = universe.universe.atoms[universe.OH_ndx].positions / a0
        dists = universe.dists[:, ::4]
    universe.ase.cell = universe.universe.dimensions[:3] / a0
    acsfs_oxy = universe.acsf_oxy.create(universe.ase, universe.oxy_ndx)
    acsfs_hyd = universe.acsf_hyd.create(universe.ase, universe.hyd_ndx)
    dists[universe.oxy_mask] = np.inf    
    nearest_ndx = np.argmin(dists, axis=1) 
    # ACSF order: interest H, bonded O, other H, nearest O, H from nearest O
    stretch_acsfs = np.empty((universe.nstretch, input_size)) 
    stretch_acsfs[:, :50] = acsfs_hyd 
    stretch_acsfs[:, 50:103] = acsfs_oxy[universe.oxy_mask[1]]
    stretch_acsfs[:, 103:153] = acsfs_hyd[universe.hyd_mask[1]]
    stretch_acsfs[:, 153:206] = acsfs_oxy[nearest_ndx]
    stretch_acsfs[:, 206:256] = acsfs_hyd[2*nearest_ndx]
    stretch_acsfs[:, 256:] = acsfs_hyd[(2*nearest_ndx)+1]

    scaled_acsfs = universe.x_scaler.transform(stretch_acsfs)
    corr_scaled = universe.model.predict(scaled_acsfs, verbose=0)
    corrections = universe.y_scaler.inverse_transform(corr_scaled)
    w_corr = w + corrections[:, 0]
    mu_corr = mu + corrections[:, 1]

 
    return w_corr, mu_corr

def switching_function(z, slab_center=0.0, r_c=4.0):
    z_adj = z - slab_center
    if hasattr(z_adj, '__len__'):
        f_z = (2*r_c**3 + 3*r_c**2 * z_adj - z_adj**3)/(4*r_c**3)
        f_z = np.where(z_adj > r_c, 1.0, f_z)
        f_z = np.where(z_adj < -r_c, 0.0, f_z)
    else:
        if z_adj > r_c:
            f_z = 1
        elif z_adj < -r_c:
            f_z = 0
        else:
            f_z = (2*r_c**3 + 3*r_c**2 * z_adj - z_adj**3)/(4*r_c**3)
    return f_z

def main():
    # Default on macOS and Windows but not on Linux
    set_start_method('spawn')

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-top', '--topology_file', required=True, metavar='FILENAME', help='Topology file for MD simulation containing water')
    parser.add_argument('-trj', '--trajectory_file', metavar='FILENAME', help='Trajectory file for MD simulation containing water')
    parser.add_argument('-s', '--start_frame', default=0, type=int, metavar='INT', help='The first frame of the trajectory to use')
    parser.add_argument('-e', '--end_frame', default=None, type=int, metavar='INT', help='The final frame of the trajectory to use')
    parser.add_argument('-n', '--n_procs', default=8, type=int, metavar='INT', help='The number of parallel processes to run')
    parser.add_argument('-f', '--ham_file', default='hamil.bin', metavar='FILENAME', help='The output Hamiltonian trajectory', type=lambda x : None if x == 'None' else x)
    parser.add_argument('-d', '--dip_file', default='dipole.bin', metavar='FILENAME', help='The output transition dipole trajectory', type=lambda x : None if x == 'None' else x)
    parser.add_argument('-r', '--ram_file', default='raman.bin', metavar='FILENAME', help='The output transition polarizability trajectory', type=lambda x : None if x == 'None' else x)
    parser.add_argument('-sfg', '--sfg_file', default=None, metavar='FILENAME', help='Transition dipole trajecotory scaled for calculating SFG spectra', type=lambda x : None if x == 'None' else x)
    parser.add_argument('-i', '--interface_axis', default='z', metavar='{x, y, z}', help='Axis perpendicular to interface for interfacial simulations')
    parser.add_argument('--fermi', action='store_true', help='Turns on Fermi resonance with bend overtone')
    parser.add_argument('-w', '--water_model', default='TIP4P', metavar='{TIP4P, E3B2, TIP3P}', help='The water model used to run the simulation')
    parser.add_argument('-wn', '--water_name', default='SOL', metavar='STR', help='The name given to water molecules in simulation files')
    parser.add_argument('-b', '--block_size', default=None, type=int, metavar='INT', help='The number of frames to calculate between writing to output files and clearing the Tensorflow session')
    parser.add_argument('-m', '--model_file', default=None, metavar='FILENAME', help='Saved TensorFlow model for using ∆-ML spectroscopic maps', type=lambda x : None if x == 'None' else x)
    parser.add_argument('-x', '--x_scaler', default='x_scaler_long_acsf.pkl', metavar='FILENAME', help='Pickled scikit-learn scaler for scaling ∆-ML model inputs', type=lambda x : None if x == 'None' else x)
    parser.add_argument('-y', '--y_scaler', default='y_scaler_acsf.pkl', metavar='FILENAME', help='Pickled scikit-learn scaler for scaling ∆-ML model outputs', type=lambda x : None if x == 'None' else x)

    
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
