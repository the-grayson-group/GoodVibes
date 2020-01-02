#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

"""####################################################################
#                              GoodVibes.py                           #
#  Evaluation of quasi-harmonic thermochemistry from Gaussian.        #
#  Partion functions are evaluated from vibrational frequencies       #
#  and rotational temperatures from the standard output.              #
#######################################################################
#  The rigid-rotor harmonic oscillator approximation is used as       #
#  standard for all frequencies above a cut-off value. Below this,    #
#  two treatments can be applied to entropic values:                  #
#    (a) low frequencies are shifted to the cut-off value (as per     #
#    Cramer-Truhlar)                                                  #
#    (b) a free-rotor approximation is applied below the cut-off (as  #
#    per Grimme). In this approach, a damping function interpolates   #
#    between the RRHO and free-rotor entropy treatment of Svib to     #
#    avoid a discontinuity.                                           #
#  Both approaches avoid infinitely large values of Svib as wave-     #
#  numbers tend to zero. With a cut-off set to 0, the results will be #
#  identical to standard values output by the Gaussian program.       #
#######################################################################
#  Enthalpy values below the cutoff value are treated similarly to    #
#  Grimme's method (as per Head-Gordon) where below the cutoff value, #
#  a damping function is applied as the value approaches a value of   #
#  0.5RT, approprate for zeolitic systems                             #
#######################################################################
#  The free energy can be evaluated for variable temperature,         #
#  concentration, vibrational scaling factor, and with a haptic       #
#  correction of the translational entropy in different solvents,     #
#  according to the amount of free space available.                   #
#######################################################################
#  A potential energy surface may be evaluated for a given set of     #
#  structures or conformers, in which case a correction to the free-  #
#  energy due to multiple conformers is applied.                      #
#  Enantiomeric excess, diastereomeric ratios and ddG can also be     #
#  calculated to show preference of stereoisomers.                    #
#######################################################################
#  Careful checks may be applied to compare variables between         #
#  multiple files such as Gaussian version, solvation models, levels  #
#  of theory, charge and multiplicity, potential duplicate structures #
#  errors in potentail linear molecules, correct or incorrect         #
#  transition states, and empirical dispersion models.                #
#######################################################################


#######################################################################
###########  Authors:     Rob Paton, Ignacio Funes-Ardoiz  ############
###########               Guilian Luchini, Juan V. Alegre- ############
###########               Requena, Yanfei Guan             ############
###########  Last modified:  July 22, 2019                 ############
####################################################################"""

import ctypes, math, os.path, sys, time
from datetime import datetime, timedelta
from glob import glob
from argparse import ArgumentParser
import numpy as np

# Importing regardless of relative import
try: from .cclib.io import ccread, ccopen
except: from cclib.io import ccread, ccopen
try: from .vib_scale_factors import scaling_data_dict, scaling_data_dict_mod, scaling_refs
except: from vib_scale_factors import scaling_data_dict, scaling_data_dict_mod, scaling_refs
try: from dftd3 import dftd3 as D3
except: pass

# VERSION NUMBER
__version__ = "3.0.2"
SUPPORTED_EXTENSIONS = set(('.out', '.log'))

# PHYSICAL CONSTANTS / UNITS
GAS_CONSTANT = 8.3144621  # J / K / mol
PLANCK_CONSTANT = 6.62606957e-34  # J * s
BOLTZMANN_CONSTANT = 1.3806488e-23  # J / K
SPEED_OF_LIGHT = 2.99792458e10  # cm / s
AVOGADRO_CONSTANT = 6.0221415e23  # 1 / mol
AMU_to_KG = 1.66053886E-27  # UNIT CONVERSION
ATMOS = 101.325  # UNIT CONVERSION
J_TO_AU = 4.184 * 627.509541 * 1000.0  # UNIT CONVERSION
KCAL_TO_AU = 627.509541  # UNIT CONVERSION

# Some literature references
grimme_ref = "Grimme, S. Chem. Eur. J. 2012, 18, 9955-9964"
truhlar_ref = "Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. J. Phys. Chem. B 2011, 115, 14556-14562"
head_gordon_ref = "Li, Y.; Gomes, J.; Sharada, S. M.; Bell, A. T.; Head-Gordon, M. J. Phys. Chem. C 2015, 119, 1840-1850"
goodvibes_ref = ("Luchini, G.; Alegre-Requena J. V.; Guan, Y.; Funes-Ardoiz, I.; Paton, R. S. (2019)."
                 "\n   GoodVibes: GoodVibes " + __version__ + " http://doi.org/10.5281/zenodo.595246")
csd_ref = ("C. R. Groom, I. J. Bruno, M. P. Lightfoot and S. C. Ward, Acta Cryst. 2016, B72, 171-179"
           "\n   Cordero, B.; Gomez V.; Platero-Prats, A. E.; Reves, M.; Echeverria, J.; Cremades, E.; Barragan, F.; Alvarez, S. Dalton Trans. 2008, 2832-2838")
oniom_scale_ref = "Simon, L.; Paton, R. S. J. Am. Chem. Soc. 2018, 140, 5412-5420"
d3_ref = "Grimme, S.; Atony, J.; Ehrlich S.; Krieg, H. J. Chem. Phys. 2010, 132, 154104"
d3bj_ref = "Grimme S.; Ehrlich, S.; Goerigk, L. J. Comput. Chem. 2011, 32, 1456-1465"
atm_ref = "Axilrod, B. M.; Teller, E. J. Chem. Phys. 1943, 11, 299 \n   Muto, Y. Proc. Phys. Math. Soc. Jpn. 1944, 17, 629"

# Some useful arrays
periodictable = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
                 "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                 "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
                 "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
                 "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
                 "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
                 "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
                 "Rg", "Uub", "Uut", "Uuq", "Uup", "Uuh", "Uus", "Uuo"]

# Symmetry numbers for different point groups
pg_sm = {"C1": 1, "Cs": 1, "Ci": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7, "C8": 8, "D2": 4, "D3": 6,
         "D4": 8, "D5": 10, "D6": 12, "D7": 14, "D8": 16, "C2v": 2, "C3v": 3, "C4v": 4, "C5v": 5, "C6v": 6, "C7v": 7,
         "C8v": 8, "C2h": 2, "C3h": 3, "C4h": 4, "C5h": 5, "C6h": 6, "C7h": 7, "C8h": 8, "D2h": 4, "D3h": 6, "D4h": 8,
         "D5h": 10, "D6h": 12, "D7h": 14, "D8h": 16, "D2d": 4, "D3d": 6, "D4d": 8, "D5d": 10, "D6d": 12, "D7d": 14,
         "D8d": 16, "S4": 4, "S6": 6, "S8": 8, "T": 6, "Th": 12, "Td": 12, "O": 12, "Oh": 24, "Cinfv": 1, "Dinfh": 2,
         "I": 30, "Ih": 60, "Kh": 1}

# Radii used to determine connectivity in symmetry corrections
# Covalent radii taken from Cambridge Structural Database
RADII = {'H': 0.32, 'He': 0.93, 'Li': 1.23, 'Be': 0.90, 'B': 0.82, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.72,
         'Ne': 0.71, 'Na': 1.54, 'Mg': 1.36, 'Al': 1.18, 'Si': 1.11, 'P': 1.06, 'S': 1.02, 'Cl': 0.99, 'Ar': 0.98,
         'K': 2.03, 'Ca': 1.74, 'Sc': 1.44, 'Ti': 1.32, 'V': 1.22, 'Cr': 1.18, 'Mn': 1.17, 'Fe': 1.17, 'Co': 1.16,
         'Ni': 1.15, 'Cu': 1.17, 'Zn': 1.25, 'Ga': 1.26, 'Ge': 1.22, 'As': 1.20, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.12,
         'Rb': 2.16, 'Sr': 1.91, 'Y': 1.62, 'Zr': 1.45, 'Nb': 1.34, 'Mo': 1.30, 'Tc': 1.27, 'Ru': 1.25, 'Rh': 1.25,
         'Pd': 1.28, 'Ag': 1.34, 'Cd': 1.48, 'In': 1.44, 'Sn': 1.41, 'Sb': 1.40, 'Te': 1.36, 'I': 1.33, 'Xe': 1.31,
         'Cs': 2.35, 'Ba': 1.98, 'La': 1.69, 'Lu': 1.60, 'Hf': 1.44, 'Ta': 1.34, 'W': 1.30, 'Re': 1.28, 'Os': 1.26,
         'Ir': 1.27, 'Pt': 1.30, 'Au': 1.34, 'Hg': 1.49, 'Tl': 1.48, 'Pb': 1.47, 'Bi': 1.46, 'X': 0}

class getoutData:
    def __init__(self, file):
        with open(file) as f:
            data = f.readlines()
        program = 'none'

        for line in data:
            if "Gaussian" in line:
                program = "Gaussian"
                break
            if "* O   R   C   A *" in line:
                program = "Orca"
                break

        def get_freqs(self, outlines, natoms, format):
            self.FREQS = []
            self.REDMASS = []
            self.FORCECONST = []
            self.NORMALMODE = []
            freqs_so_far = 0
            if format == "Gaussian":
                for i in range(0, len(outlines)):
                    if outlines[i].find(" Frequencies -- ") > -1:
                        nfreqs = len(outlines[i].split())
                        for j in range(2, nfreqs):
                            self.FREQS.append(float(outlines[i].split()[j]))
                            self.NORMALMODE.append([])
                        for j in range(3, nfreqs + 1): self.REDMASS.append(float(outlines[i + 1].split()[j]))
                        for j in range(3, nfreqs + 1): self.FORCECONST.append(float(outlines[i + 2].split()[j]))

                        for j in range(0, natoms):
                            for k in range(0, nfreqs - 2):
                                self.NORMALMODE[(freqs_so_far + k)].append(
                                    [float(outlines[i + 5 + j].split()[3 * k + 2]),
                                     float(outlines[i + 5 + j].split()[3 * k + 3]),
                                     float(outlines[i + 5 + j].split()[3 * k + 4])])
                        freqs_so_far = freqs_so_far + nfreqs - 2

        def getatom_types(self, outlines, program):
            if program == "Gaussian":
                for i, oline in enumerate(outlines):
                    if "Input orientation" in oline or "Standard orientation" in oline:
                        self.atom_nums, self.atom_types, self.cartesians, self.atomictypes, carts = [], [], [], [], \
                                                                                                    outlines[i + 5:]
                        for j, line in enumerate(carts):
                            if "-------" in line:
                                break
                            self.atom_nums.append(int(line.split()[1]))
                            self.atom_types.append(element_id(int(line.split()[1])))
                            self.atomictypes.append(int(line.split()[2]))
                            if len(line.split()) > 5:
                                self.cartesians.append(
                                    [float(line.split()[3]), float(line.split()[4]), float(line.split()[5])])
                            else:
                                self.cartesians.append(
                                    [float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
            if program == "Orca":
                for i, oline in enumerate(outlines):
                    if "*" in oline and ">" in oline and "xyz" in oline:
                        self.atom_nums, self.atom_types, self.cartesians, carts = [], [], [], outlines[i + 1:]
                        for j, line in enumerate(carts):
                            if ">" in line and "*" in line:
                                break
                            if len(line.split()) > 5:
                                self.cartesians.append(
                                    [float(line.split()[3]), float(line.split()[4]), float(line.split()[5])])
                                self.atom_types.append(line.split()[2])
                                self.atom_nums.append(element_id(line.split()[2], num=True))
                            else:
                                self.cartesians.append(
                                    [float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
                                self.atom_types.append(line.split()[1])
                                self.atom_nums.append(element_id(line.split()[1], num=True))

        getatom_types(self, data, program)
        natoms = len(self.atom_types)
        try:
            get_freqs(self, data, natoms, program)
        except:
            pass

    # Convert coordinates to string that can be used by the symmetry.c program
    def coords_string(self):
        xyzstring = str(len(self.atom_nums)) + '\n'
        for atom, xyz in zip(self.atom_nums, self.cartesians):
            xyzstring += "{0} {1:.6f} {2:.6f} {3:.6f}\n".format(atom, *xyz)
        return xyzstring

    # Obtain molecule connectivity to be used for internal symmetry determination
    def get_connectivity(self):
        connectivity = []
        tolerance = 0.2

        for i, ai in enumerate(self.atom_types):
            row = []
            for j, aj in enumerate(self.atom_types):
                if i == j:
                    continue
                cutoff = RADII[ai] + RADII[aj] + tolerance
                distance = np.linalg.norm(np.array(self.cartesians[i]) - np.array(self.cartesians[j]))
                if distance < cutoff:
                    row.append(j)
            connectivity.append(row)
            self.connectivity = connectivity

def sharepath(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'share', filename)

def element_id(massno, num=False):
    try:
        if num:
            return periodictable.index(massno)
        return periodictable[massno]
    except IndexError:
        return "XX"

def all_same(items):
    return all(x == items[0] for x in items)

alphabet = 'abcdefghijklmnopqrstuvwxyz'

# Enables output to terminal and to text file
class Logger:
    def __init__(self, filein, append, csv):
        self.csv = csv
        if not self.csv:
            suffix = 'dat'
        else:
            suffix = 'csv'
        self.log = open('{0}_{1}.{2}'.format(filein, append, suffix), 'w')

    def write(self, message, thermodata=False):
        self.thermodata = thermodata
        print(message, end='')
        if self.csv and self.thermodata:
            items = message.split()
            message = ",".join(items)
            message = message + ","
        self.log.write(message)

    def fatal(self, message):
        print(message + "\n")
        self.log.write(message + "\n")
        self.finalize()
        sys.exit(1)

    def finalize(self):
        self.log.close()

# Calculate elapsed time
def add_time(tm, cpu):
    [days, hrs, mins, secs, msecs] = cpu
    fulldate = datetime(100, 1, tm.day, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(days=days, hours=hrs, minutes=mins, seconds=secs, microseconds=msecs * 1000)
    return fulldate

def calc_cpu(files, thermo_data, options, log):
    # Initialize the total CPU time
    add_days = 0
    cpu = datetime(100, 1, 1, 00, 00, 00, 00)
    for file in files:
        bbe = thermo_data[file]
        if options.cputime != False:  # Add up CPU times
            if hasattr(bbe, "cpu"):
                if bbe.cpu != None:
                    cpu = add_time(cpu, bbe.cpu)

    if cpu.month > 1: add_days += 31 * (cpu.month -1)
    else: add_days = 0
    log.write('   {:<13} {:>2} {:>4} {:>2} {:>3} {:>2} {:>4} {:>2} '
              '{:>4}\n'.format('TOTAL CPU', cpu.day + add_days - 1, 'days', cpu.hour, 'hrs',
                               cpu.minute, 'mins', cpu.second, 'secs'))
    return cpu

def coords_string(self):
    xyzstring = str(len(self.atom_nums)) + '\n'
    for atom, xyz in zip(self.atom_nums, self.cartesians):
        xyzstring += "{0} {1:.6f} {2:.6f} {3:.6f}\n".format(atom, *xyz)
    return xyzstring

def get_connectivity(self):

    connectivity = []
    tolerance = 0.2

    for i, ai in enumerate(self.atom_types):
        row = []
        for j, aj in enumerate(self.atom_types):
            if i == j:
                continue
            cutoff = RADII[ai] + RADII[aj] + tolerance
            distance = np.linalg.norm(np.array(self.cartesians[i]) - np.array(self.cartesians[j]))
            if distance < cutoff:
                row.append(j)
        connectivity.append(row)
        self.connectivity = connectivity

# Enables output of optimized coordinates to a single xyz-formatted file
class xyz_out:
    ''' writes multiple structures out to xyz format'''
    def __init__(self, xyz_file, thermo_data):

        self.xyz = open(xyz_file, 'w')

        for file in thermo_data:
            if hasattr(file, 'natom'): self.xyz.write(str(file.natom)+"\n")
            if hasattr(file, "scfenergies"):
                self.xyz.write(
                    '{:<39} {:>13} {:13.6f}\n'.format(os.path.splitext(os.path.basename(file.name))[0], 'Eopt',
                                                    file.scfenergies[-1]))
            else:
                self.xyz.write('{:<39}\n'.format(os.path.splitext(os.path.basename(file.name))[0]))
            if hasattr(file, 'atomcoords') and hasattr(file, 'atomnos'):
                for n, atom in enumerate(file.atomnos):
                    self.xyz.write('{:>1}'.format(periodictable[int(atom)]))
                    for cart in file.atomcoords[-1][n]:
                        self.xyz.write('{:13.6f}'.format(cart))
                    self.xyz.write('\n')

        self.xyz.close()

# The function to compute the "black box" entropy and enthalpy values
# along with all other thermochemical quantities
class calc_bbe:
    def __init__(self, file, options, ssymm=False, cosmo=None, mm_freq_scale_factor=False):
        ''' the thermochemistry calculation using quasi RRHO'''

        # Careful with single atoms!
        if file.natom == 1:
            file.rotemp, file.roconst, file.vibfreqs = [], [], []

        if not hasattr(file, 'rotemp'): print('\nx  Missing rotemp in ', file.name)
        if not hasattr(file, 'roconst'): print('x  Missing roconst in ', file.name)
        else: self.roconst = file.roconst
        if not hasattr(file, 'symmno'): print('x  Missing symmnoin ', file.name)
        if not hasattr(file, 'cpu'): print('x  Missing cpu in ', file.name)
        if not hasattr(file, 'linear_mol'): print('x  Missing linear_mol in ', file.name)
        if not hasattr(file, 'vibfreqs'): print('x  Missing vibfreqs in ', file.name)
        if not hasattr(file, 'atomcoords'): print('x  Missing atomcoords in ', file.name)
        if not hasattr(file, 'point_group'): print('x  Missing pointgroup in ', file.name)

        if options.spc is not False:
            cc_data, kwargs = None, {}
            for spc_file in [file.name+'_'+options.spc+'.out', file.name+'_'+options.spc+'.log', file.name+'-'+options.spc+'.out', file.name+'-'+options.spc+'.log']:
                if os.path.exists(spc_file):
                    cc_data = ccread(spc_file, **kwargs)

            if hasattr(cc_data, 'scfenergies'): self.sp_energy = cc_data.scfenergies[-1]
            if hasattr(cc_data, 'single_point_energy'): self.sp_energy = cc_data.single_point_energy
            if hasattr(cc_data, 'cpu'): self.sp_cpu = cc_data.cpu

            if cc_data == None:
                print('\nx  Missing spc data for', file.name, end='')
                self.sp_energy = '!'

        if options.solv is not False:
            cc_data, kwargs = None, {}
            for solv_file in [file.name+'_'+options.solv+'.out', file.name+'_'+options.solv+'.log', file.name+'-'+options.solv+'.out', file.name+'-'+options.solv+'.log']:
                if os.path.exists(solv_file):
                    cc_data = ccread(solv_file, **kwargs)

            self.dg_solv = 0.0
            if hasattr(cc_data, 'dgsolv'): self.dg_solv = cc_data.dgsolv

            if cc_data == None:
                print('\nx  Missing solvation data for', file.name, end='')


        frequency_wn, im_frequency_wn, inverted_freqs = [], [], []

        if hasattr(file, 'cpu'): self.cpu = file.cpu
        else: self.cpu = [0,0,0,0,0]
        # adds the time spend doing single point calculation to total
        if hasattr(self, 'sp_cpu'): self.cpu = [cpu + sp_cpu for cpu, sp_cpu in zip(self.cpu, self.sp_cpu)]

        if hasattr(file, 'vibfreqs'):
            for freq in file.vibfreqs:
                if freq > 0.00:
                    frequency_wn.append(freq)
                elif freq < 0.00:
                    if options.invert is not False:
                        if abs(freq) < abs(float(options.invert)):
                            frequency_wn.append(freq * -1.0)
                            inverted_freqs.append(freq)
                        else: im_frequency_wn.append(freq)
                    else: im_frequency_wn.append(freq)

        linear_warning = False

        if mm_freq_scale_factor is False:
            fract_modelsys = False
        else:
            fract_modelsys = []
            freq_scale_factor = [freq_scale_factor, mm_freq_scale_factor]

        self.inverted_freqs = inverted_freqs

        # Symmetry - entropy correction for molecular symmetry
        if options.ssymm:
            '''This could all be sped up using file.atomcoords and file.atomnos which we have already at this point
            without reading the file in again. It would avoid the need for the parsing aspects of getoutData '''
            self.xyz = getoutData(file.name+'.log')
            file.symmno, self.point_group = self.ex_sym(file.name.split('.')[0])
            file.int_sym = self.int_sym()

        # override internal symmetry
        file.int_sym = 1

        # Skip the calculation if unable to parse the output file
        if hasattr(file, 'molecular_mass') and hasattr(file, 'mult'):

            cutoffs = [options.S_freq_cutoff for freq in frequency_wn]

            # Translational and electronic contributions to the energy and entropy do not depend on frequencies
            u_trans = calc_translational_energy(options.temperature)
            s_trans = calc_translational_entropy(file.molecular_mass, options.conc, options.temperature, options.freespace)
            s_elec = calc_electronic_entropy(file.mult)

        if hasattr(file, 'vibfreqs') and hasattr(file, 'rotemp') and hasattr(file, 'symmno') and hasattr(file, 'linear_mol'):
            # Rotational and Vibrational contributions to the energy entropy
            if len(frequency_wn) > 0:
                zpe = calc_zeropoint_energy(frequency_wn, options.freq_scale_factor, fract_modelsys)
                u_rot = calc_rotational_energy(zpe, file.symmno, options.temperature, file.linear_mol)
                u_vib = calc_vibrational_energy(frequency_wn, options.temperature, options.freq_scale_factor, fract_modelsys)
                s_rot = calc_rotational_entropy(zpe, file.linear_mol, file.symmno, file.rotemp, options.temperature)

                # Calculate harmonic entropy, free-rotor entropy and damping function for each frequency
                Svib_rrho = calc_rrho_entropy(frequency_wn, options.temperature, options.freq_scale_factor, fract_modelsys)

                if options.S_freq_cutoff > 0.0:
                    Svib_rrqho = calc_rrho_entropy(cutoffs, options.temperature, options.freq_scale_factor, fract_modelsys)
                Svib_free_rot = calc_freerot_entropy(frequency_wn, options.temperature, options.freq_scale_factor, fract_modelsys, file.int_sym)
                S_damp = calc_damp(frequency_wn, options.S_freq_cutoff)

                # check for qh
                if options.QH:
                    Uvib_qrrho = calc_qRRHO_energy(frequency_wn, options.temperature, options.freq_scale_factor)
                    H_damp = calc_damp(frequency_wn, options.H_freq_cutoff)

                # Compute entropy (cal/mol/K) using the two values and damping function
                vib_entropy, vib_energy = [], []

                for j in range(0, len(frequency_wn)):
                    # Entropy correction
                    if options.QS == "grimme":
                        vib_entropy.append(Svib_rrho[j] * S_damp[j] + (1 - S_damp[j]) * Svib_free_rot[j])
                    elif options.QS == "truhlar":
                        if options.S_freq_cutoff > 0.0:
                            if frequency_wn[j] > options.S_freq_cutoff:
                                vib_entropy.append(Svib_rrho[j])
                            else:
                                vib_entropy.append(Svib_rrqho[j])
                        else:
                            vib_entropy.append(Svib_rrho[j])
                    # Enthalpy correction
                    if options.QH:
                        vib_energy.append(H_damp[j] * Uvib_qrrho[j] + (1 - H_damp[j]) * 0.5 * GAS_CONSTANT * options.temperature)

                qh_s_vib, h_s_vib = sum(vib_entropy), sum(Svib_rrho)

                if options.QH:
                    qh_u_vib = sum(vib_energy)
            else:
                zpe, u_rot, u_vib, qh_u_vib, s_rot, h_s_vib, qh_s_vib = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # electronic energy term
            self.scf_energy = 0.0
            if hasattr(file, 'scfenergies'): self.scf_energy += file.scfenergies[-1]

            # The D3 term is added to the energy term here. If not requested then this term is zero
            # It is added to the SPC energy if defined (instead of the SCF energy)
            # computes D3 term if requested, which is then sent to calc_bbe as a correction
            if options.D3 or options.D3BJ:
                verbose, intermolecular, pairwise, abc_term = False, False, False, False
                s6, rs6, s8, bj_a1, bj_a2 = 0.0, 0.0, 0.0, 0.0, 0.0

                try: functional = file.metadata["functional"]
                except: functional = None

                if options.D3: damp = 'zero'
                elif options.D3BJ: damp = 'bj'
                if options.ATM: abc_term = True

                try:
                    d3_calc = D3.calcD3(file, functional, s6, rs6, s8, bj_a1, bj_a2, damp, abc_term, intermolecular,
                                        pairwise, verbose)
                    if options.ATM: d3_term = (d3_calc.attractive_r6_vdw + d3_calc.attractive_r8_vdw + d3_calc.repulsive_abc) / KCAL_TO_AU
                    else: d3_term = (d3_calc.attractive_r6_vdw + d3_calc.attractive_r8_vdw) / KCAL_TO_AU
                except:
                    print('   ! Dispersion Correction Failed for {}'.format(file.name))
                    d3_term = 0.0

                if options.spc is False:
                    self.scf_energy += d3_term
                elif hasattr(self, "sp_energy"):
                    if self.sp_energy != '!':
                        self.sp_energy += d3_term

            # Add terms (converted to au) to get Free energy - perform separately
            # for harmonic and quasi-harmonic values out of interest
            self.enthalpy = self.scf_energy + (u_trans + u_rot + u_vib + GAS_CONSTANT * options.temperature) / J_TO_AU

            if options.QH:
                self.qh_enthalpy = self.scf_energy + (u_trans + u_rot + qh_u_vib + GAS_CONSTANT * options.temperature) / J_TO_AU
            else: self.qh_enthalpy = 0.0

            # Single point correction replaces energy from optimization with single point value
            if options.spc is not False:
                if hasattr(self, "sp_energy"):
                    if self.sp_energy != '!':
                        try:
                            self.enthalpy = self.enthalpy - self.scf_energy + self.sp_energy
                        except TypeError:
                            pass
                        if options.QH:
                            try:
                                self.qh_enthalpy = self.qh_enthalpy - self.scf_energy + self.sp_energy
                            except TypeError:
                                pass

            self.zpe = zpe / J_TO_AU
            self.entropy = (s_trans + s_rot + h_s_vib + s_elec) / J_TO_AU
            self.qh_entropy = (s_trans + s_rot + qh_s_vib + s_elec) / J_TO_AU

            # Calculate Free Energy
            if options.QH:
                self.gibbs_free_energy = self.enthalpy - options.temperature * self.entropy
                self.qh_gibbs_free_energy = self.qh_enthalpy - options.temperature * self.qh_entropy
            else:
                self.gibbs_free_energy = self.enthalpy - options.temperature * self.entropy
                self.qh_gibbs_free_energy = self.enthalpy - options.temperature * self.qh_entropy

            if options.solv is not False:
                self.solv_qhg = self.qh_gibbs_free_energy + self.dg_solv / KCAL_TO_AU
            elif options.cosmo or options.solv:
                self.solv_qhg = self.qh_gibbs_free_energy + cosmo
            else:
                self.solv_qhg = self.qh_gibbs_free_energy

            self.im_freq = []
            for freq in im_frequency_wn:
                self.im_freq.append(freq)

        self.frequency_wn = frequency_wn
        self.im_frequency_wn = im_frequency_wn
        self.linear_warning = linear_warning

    # Get external symmetry number
    def ex_sym(self, file):
        coords_string = self.xyz.coords_string()
        coords = coords_string.encode('utf-8')
        c_coords = ctypes.c_char_p(coords)

        # Determine OS with sys.platform to see what compiled symmetry file to use
        platform = sys.platform
        if platform.startswith('linux'):  # linux - .so file
            path1 = sharepath('symmetry_linux.so')
            newlib = 'lib_' + file + '.so'
            path2 = sharepath(newlib)
            copy = 'cp ' + path1 + ' ' + path2
            os.popen(copy).close()
            symmetry = ctypes.CDLL(path2)
        elif platform.startswith('darwin'):  # macOS - .dylib file
            path1 = sharepath('symmetry_mac.dylib')
            newlib = 'lib_' + file + '.dylib'
            path2 = sharepath(newlib)
            copy = 'cp ' + path1 + ' ' + path2
            os.popen(copy).close()
            symmetry = ctypes.CDLL(path2)
        elif platform.startswith('win'):  # windows - .dll file
            path1 = sharepath('symmetry_windows.dll')
            newlib = 'lib_' + file + '.dll'
            path2 = sharepath(newlib)
            copy = 'copy ' + path1 + ' ' + path2
            os.popen(copy).close()
            symmetry = ctypes.cdll.LoadLibrary(path2)

        symmetry.symmetry.restype = ctypes.c_char_p
        pgroup = symmetry.symmetry(c_coords).decode('utf-8')
        ex_sym = pg_sm.get(pgroup)

        # Remove file
        if platform.startswith('linux'):  # linux - .so file
            remove = 'rm ' + path2
            os.popen(remove).close()
        elif platform.startswith('darwin'):  # macOS - .dylib file
            remove = 'rm ' + path2
            os.popen(remove).close()
        elif platform.startswith('win'):  # windows - .dll file
            handle = symmetry._handle
            del symmetry
            ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle))
            remove = 'Del /F "' + path2 + '"'
            os.popen(remove).close()

        return ex_sym, pgroup

    def int_sym(self):
        self.xyz.get_connectivity()
        cap = [1, 9, 17]
        neighbor = [5, 6, 7, 8, 14, 15, 16]
        int_sym = 1

        for i, row in enumerate(self.xyz.connectivity):
            if self.xyz.atom_nums[i] != 6: continue
            As = np.array(self.xyz.atom_nums)[row]
            if len(As == 4):
                neighbors = [x for x in As if x in neighbor]
                caps = [x for x in As if x in cap]
                if (len(neighbors) == 1) and (len(set(caps)) == 1):
                    int_sym *= 3
        return int_sym

# Obtain relative thermochemistry between species and for reactions
class get_pes:
    def __init__(self, thermo_data, options, log, cosmo=None, cosmo_int=None):

        for key in thermo_data:
            if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)
            if not hasattr(thermo_data[key], "sp_energy") and options.spc is not False:
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)

        # Default values
        self.dec, self.units, self.boltz = 2, 'kcal/mol', False

        file = options.pes

        with open(file) as f:
            data = f.readlines()
        folder, program, names, files, zeros, pes_list = None, None, [], [], [], []
        for i, dline in enumerate(data):
            if dline.strip().find('PES') > -1:
                for j, line in enumerate(data[i + 1:]):
                    if line.strip().startswith('#'):
                        pass
                    elif len(line) <= 2:
                        pass
                    elif line.strip().startswith('---'):
                        break
                    elif line.strip() != '':
                        pathway, pes = line.strip().replace(':', '=').split("=")
                        # Auto-grab first species as zero unless specified
                        pes_list.append(pes)
                        zeros.append(pes.strip().lstrip('[').rstrip(']').split(',')[0])
                        # Look at SPECIES block to determine filenames
            if dline.strip().find('SPECIES') > -1:
                for j, line in enumerate(data[i + 1:]):
                    if line.strip().startswith('---'):
                        break
                    else:
                        if line.lower().strip().find('folder') > -1:
                            try:
                                folder = line.strip().replace('#', '=').split("=")[1].strip()
                            except IndexError:
                                pass
                        else:
                            try:
                                n, f = (line.strip().replace(':', '=').split("="))
                                # Check the specified filename is also one that GoodVibes has thermochemistry for:
                                if f.find('*') == -1 and f not in pes_list:
                                    match = None
                                    for key in thermo_data:
                                        if os.path.splitext(os.path.basename(key))[0] in f.replace('[', '').replace(']', '').replace('+', ',').replace(' ', '').split(','):
                                            match = key
                                    if match:
                                        names.append(n.strip())
                                        files.append(match)
                                    else:
                                        log.write("   Warning! " + f.strip() + ' is specified in ' + file +
                                                  ' but no thermochemistry data found\n')
                                elif f not in pes_list:
                                    match = []
                                    for key in thermo_data:
                                        if os.path.splitext(os.path.basename(key))[0].find(f.strip().strip('*')) == 0:
                                            match.append(key)
                                    if len(match) > 0:
                                        names.append(n.strip())
                                        files.append(match)
                                    else:
                                        log.write("   Warning! " + f.strip() + ' is specified in ' + file +
                                                  ' but no thermochemistry data found\n')
                            except ValueError:
                                if line.isspace():
                                    pass
                                elif line.strip().find('#') > -1:
                                    pass
                                elif len(line) > 2:
                                    warn = "   Warning! " + file + ' input is incorrectly formatted for line:\n\t' + line
                                    log.write(warn)
            # Look at FORMAT block to see if user has specified any formatting rules
            if dline.strip().find('FORMAT') > -1:
                for j, line in enumerate(data[i + 1:]):
                    if line.strip().find('dec') > -1:
                        try:
                            self.dec = int(line.strip().replace(':', '=').split("=")[1].strip())
                        except IndexError:
                            pass
                    if line.strip().find('zero') > -1:
                        zeros = []
                        try:
                            zeros.append(line.strip().replace(':', '=').split("=")[1].strip())
                        except IndexError:
                            pass
                    if line.strip().find('units') > -1:
                        try:
                            self.units = line.strip().replace(':', '=').split("=")[1].strip()
                        except IndexError:
                            pass
                    if line.strip().find('boltz') > -1:
                        try:
                            self.boltz = line.strip().replace(':', '=').split("=")[1].strip()
                        except IndexError:
                            pass

        for i in range(len(files)):
            if len(files[i]) is 1:
                files[i] = files[i][0]
        species = dict(zip(names, files))
        self.path, self.species = [], []
        self.spc_abs, self.e_abs, self.zpe_abs, self.h_abs, self.qh_abs, self.s_abs, self.qs_abs, self.g_abs, self.qhg_abs, self.solv_qhg_abs = [], [], [], [], [], [], [], [], [], []
        self.spc_zero, self.e_zero, self.zpe_zero, self.h_zero, self.qh_zero, self.ts_zero, self.qhts_zero, self.g_zero, self.qhg_zero, self.solv_qhg_zero = [], [], [], [], [], [], [], [], [], []
        self.g_qhgvals, self.g_species_qhgzero, self.g_rel_val = [], [], []
        # Loop over .yaml file, grab energies, populate arrays and compute Boltzmann factors
        with open(file) as f:
            data = f.readlines()
        for i, dline in enumerate(data):
            if dline.strip().find('PES') > -1:
                n = 0
                for j, line in enumerate(data[i + 1:]):
                    if line.strip().startswith('#') == True:
                        pass
                    elif len(line) <= 2:
                        pass
                    elif line.strip().startswith('---') == True:
                        break
                    elif line.strip() != '':
                        try:
                            self.e_zero.append([])
                            self.spc_zero.append([])
                            self.zpe_zero.append([])
                            self.h_zero.append([])
                            self.qh_zero.append([])
                            self.ts_zero.append([])
                            self.qhts_zero.append([])
                            self.g_zero.append([])
                            self.qhg_zero.append([])
                            self.solv_qhg_zero.append([])
                            min_conf = False
                            spc_zero, e_zero, zpe_zero, h_zero, qh_zero, s_zero, qs_zero, g_zero, qhg_zero = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                            h_conf, h_tot, s_conf, s_tot, qh_conf, qh_tot, qs_conf, qs_tot, solv_qhg_zero = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                            zero_structures = zeros[n].replace(' ', '').split('+')
                            # Routine for 'zero' values
                            for structure in zero_structures:
                                try:
                                    if not isinstance(species[structure], list):
                                        if hasattr(thermo_data[species[structure]], "sp_energy"):
                                            if thermo_data[species[structure]].sp_energy != '!':
                                                spc_zero += thermo_data[species[structure]].sp_energy
                                            else:
                                                spc_zero += 0.0
                                        e_zero += thermo_data[species[structure]].scf_energy
                                        zpe_zero += thermo_data[species[structure]].zpe
                                        h_zero += thermo_data[species[structure]].enthalpy
                                        qh_zero += thermo_data[species[structure]].qh_enthalpy
                                        s_zero += thermo_data[species[structure]].entropy
                                        qs_zero += thermo_data[species[structure]].qh_entropy
                                        g_zero += thermo_data[species[structure]].gibbs_free_energy
                                        qhg_zero += thermo_data[species[structure]].qh_gibbs_free_energy
                                        solv_qhg_zero += thermo_data[species[structure]].solv_qhg

                                    else:  # If we have a list of different kinds of structures: loop over conformers
                                        g_min, boltz_sum = sys.float_info.max, 0.0
                                        for conformer in species[
                                            structure]:  # Find minimum G, along with associated enthalpy and entropy
                                            if cosmo or options.solv:
                                                if thermo_data[conformer].solv_qhg <= g_min:
                                                    min_conf = thermo_data[conformer]
                                                    g_min = thermo_data[conformer].solv_qhg
                                            else:
                                                if thermo_data[conformer].qh_gibbs_free_energy <= g_min:
                                                    min_conf = thermo_data[conformer]
                                                    g_min = thermo_data[conformer].qh_gibbs_free_energy
                                        for conformer in species[structure]:  # Get a Boltzmann sum for conformers
                                            if cosmo or options.solv:
                                                g_rel = thermo_data[conformer].solv_qhg - g_min
                                            else:
                                                g_rel = thermo_data[conformer].qh_gibbs_free_energy - g_min
                                            boltz_fac = math.exp(-g_rel * J_TO_AU / GAS_CONSTANT / options.temperature)
                                            boltz_sum += boltz_fac
                                        for conformer in species[
                                            structure]:  # Calculate relative data based on Gmin and the Boltzmann sum
                                            if cosmo or options.solv:
                                                g_rel = thermo_data[conformer].solv_qhg - g_min
                                            else:
                                                g_rel = thermo_data[conformer].qh_gibbs_free_energy - g_min
                                            boltz_fac = math.exp(-g_rel * J_TO_AU / GAS_CONSTANT / options.temperature)
                                            boltz_prob = boltz_fac / boltz_sum
                                            if hasattr(thermo_data[conformer], "sp_energy") and thermo_data[
                                                conformer].sp_energy is not '!':
                                                spc_zero += thermo_data[conformer].sp_energy * boltz_prob
                                            if hasattr(thermo_data[conformer], "sp_energy") and thermo_data[
                                                conformer].sp_energy is '!':
                                                sys.exit(
                                                    "Not all files contain a SPC value, relative values will not be calculated.")
                                            e_zero += thermo_data[conformer].scf_energy * boltz_prob
                                            zpe_zero += thermo_data[conformer].zpe * boltz_prob

                                            if options.gconf:  # Default calculate gconf correction for conformers
                                                h_conf += thermo_data[conformer].enthalpy * boltz_prob
                                                s_conf += thermo_data[conformer].entropy * boltz_prob
                                                s_conf += -GAS_CONSTANT / J_TO_AU * boltz_prob * math.log(boltz_prob)

                                                qh_conf += thermo_data[conformer].qh_enthalpy * boltz_prob
                                                qs_conf += thermo_data[conformer].qh_entropy * boltz_prob
                                                qs_conf += -GAS_CONSTANT / J_TO_AU * boltz_prob * math.log(boltz_prob)
                                            #else:
                                            h_zero += thermo_data[conformer].enthalpy * boltz_prob
                                            s_zero += thermo_data[conformer].entropy * boltz_prob
                                            g_zero += thermo_data[conformer].gibbs_free_energy * boltz_prob

                                            qh_zero += thermo_data[conformer].qh_enthalpy * boltz_prob
                                            qs_zero += thermo_data[conformer].qh_entropy * boltz_prob
                                            qhg_zero += thermo_data[conformer].qh_gibbs_free_energy * boltz_prob
                                            solv_qhg_zero += thermo_data[conformer].solv_qhg * boltz_prob

                                        if options.gconf:
                                            h_adj = h_conf - min_conf.enthalpy
                                            h_tot = min_conf.enthalpy + h_adj
                                            s_adj = s_conf - min_conf.entropy
                                            s_tot = min_conf.entropy + s_adj
                                            g_corr = h_tot - options.temperature * s_tot
                                            qh_adj = qh_conf - min_conf.qh_enthalpy
                                            qh_tot = min_conf.qh_enthalpy + qh_adj
                                            qs_adj = qs_conf - min_conf.qh_entropy
                                            qs_tot = min_conf.qh_entropy + qs_adj
                                            if options.QH:
                                                qg_corr = qh_tot - options.temperature * qs_tot
                                            else:
                                                qg_corr = h_tot - options.temperature * qs_tot
                                            solv_qg_corr = solv_qhg_zero + qg_corr - h_zero + options.temperature * qs_zero

                                except KeyError:
                                    log.write(
                                        "   Warning! Structure " + structure + ' has not been defined correctly as energy-zero in ' + file + '\n')
                                    log.write(
                                        "   Make sure this structure matches one of the SPECIES defined in the same file\n")
                                    sys.exit("   Please edit " + file + " and try again\n")
                            # Set zero vals here
                            conformers, single_structure, mix = False, False, False
                            for structure in zero_structures:
                                if not isinstance(species[structure], list):
                                    single_structure = True
                                else:
                                    conformers = True
                            if conformers and single_structure:
                                mix = True

                            if options.gconf and min_conf is not False:
                                if mix:
                                    h_mix = h_tot + h_zero
                                    s_mix = s_tot + s_zero
                                    g_mix = g_corr + g_zero
                                    qh_mix = qh_tot + qh_zero
                                    qs_mix = qs_tot + qs_zero
                                    qg_mix = qg_corr + qhg_zero
                                    solv_qhg_mix = qg_corr + solv_qhg_zero
                                    self.h_zero[n].append(h_mix)
                                    self.ts_zero[n].append(s_mix)
                                    self.g_zero[n].append(g_mix)
                                    self.qh_zero[n].append(qh_mix)
                                    self.qhts_zero[n].append(qs_mix)
                                    self.qhg_zero[n].append(qg_mix)
                                    self.solv_qhg_zero[n].append(solv_qhg_mix)
                                elif conformers:
                                    self.h_zero[n].append(h_tot)
                                    self.ts_zero[n].append(s_tot)
                                    self.g_zero[n].append(g_corr)
                                    self.qh_zero[n].append(qh_tot)
                                    self.qhts_zero[n].append(qs_tot)
                                    self.qhg_zero[n].append(qg_corr)
                                    self.solv_qhg_zero[n].append(solv_qg_corr)
                            else:
                                self.h_zero[n].append(h_zero)
                                self.ts_zero[n].append(s_zero)
                                self.g_zero[n].append(g_zero)

                                self.qh_zero[n].append(qh_zero)
                                self.qhts_zero[n].append(qs_zero)
                                self.qhg_zero[n].append(qhg_zero)
                                self.solv_qhg_zero[n].append(solv_qhg_zero)

                            self.spc_zero[n].append(spc_zero)
                            self.e_zero[n].append(e_zero)
                            self.zpe_zero[n].append(zpe_zero)

                            self.species.append([])
                            self.e_abs.append([])
                            self.spc_abs.append([])
                            self.zpe_abs.append([])
                            self.h_abs.append([])
                            self.qh_abs.append([])
                            self.s_abs.append([])
                            self.g_abs.append([])
                            self.qs_abs.append([])
                            self.qhg_abs.append([])
                            self.solv_qhg_abs.append([])
                            self.g_qhgvals.append([])
                            self.g_species_qhgzero.append([])
                            self.g_rel_val.append([])  # graphing

                            pathway, pes = line.strip().replace(':', '=').split("=")
                            pes = pes.strip()
                            points = [entry.strip() for entry in pes.lstrip('[').rstrip(']').split(',')]
                            self.path.append(pathway.strip())
                            # Obtain relative values for each species
                            for i, point in enumerate(points):
                                if point != '':
                                    # Create values to populate
                                    point_structures = point.replace(' ', '').split('+')
                                    e_abs, spc_abs, zpe_abs, h_abs, qh_abs, s_abs, g_abs, qs_abs, qhg_abs, solv_qhg_abs = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                                    qh_conf, qh_tot, qs_conf, qs_tot, h_conf, h_tot, s_conf, s_tot, g_corr, qg_corr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                                    min_conf = False
                                    rel_val = 0.0
                                    self.g_qhgvals[n].append([])
                                    self.g_species_qhgzero[n].append([])
                                    try:
                                        for j, structure in enumerate(point_structures):  # Loop over structures, structures are species specified
                                            zero_conf = 0.0
                                            self.g_qhgvals[n][i].append([])
                                            if not isinstance(species[structure], list):  # Only one conf in structures
                                                e_abs += thermo_data[species[structure]].scf_energy
                                                if hasattr(thermo_data[species[structure]], "sp_energy"):
                                                    if thermo_data[species[structure]].sp_energy != '!':
                                                        spc_abs += thermo_data[species[structure]].sp_energy
                                                    else:
                                                        spc_abs += 0.0
                                                zpe_abs += thermo_data[species[structure]].zpe
                                                h_abs += thermo_data[species[structure]].enthalpy
                                                qh_abs += thermo_data[species[structure]].qh_enthalpy
                                                s_abs += thermo_data[species[structure]].entropy
                                                g_abs += thermo_data[species[structure]].gibbs_free_energy
                                                qs_abs += thermo_data[species[structure]].qh_entropy
                                                qhg_abs += thermo_data[species[structure]].qh_gibbs_free_energy
                                                solv_qhg_abs += thermo_data[species[structure]].solv_qhg
                                                zero_conf += thermo_data[species[structure]].qh_gibbs_free_energy
                                                self.g_qhgvals[n][i][j].append(
                                                    thermo_data[species[structure]].qh_gibbs_free_energy)
                                                rel_val += thermo_data[species[structure]].qh_gibbs_free_energy
                                            else:  # If we have a list of different kinds of structures: loop over conformers
                                                g_min, boltz_sum = sys.float_info.max, 0.0
                                                # Find minimum G, along with associated enthalpy and entropy
                                                for conformer in species[structure]:
                                                    if cosmo or options.solv:
                                                        if thermo_data[conformer].solv_qhg <= g_min:
                                                            min_conf = thermo_data[conformer]
                                                            g_min = thermo_data[conformer].solv_qhg
                                                    else:
                                                        if thermo_data[conformer].qh_gibbs_free_energy <= g_min:
                                                            min_conf = thermo_data[conformer]
                                                            g_min = thermo_data[conformer].qh_gibbs_free_energy
                                                # Get a Boltzmann sum for conformers
                                                for conformer in species[structure]:
                                                    if cosmo or options.solv:
                                                        g_rel = thermo_data[conformer].solv_qhg - g_min
                                                    else:
                                                        g_rel = thermo_data[conformer].qh_gibbs_free_energy - g_min
                                                    boltz_fac = math.exp(-g_rel * J_TO_AU / GAS_CONSTANT / options.temperature)
                                                    boltz_sum += boltz_fac
                                                # Calculate relative data based on Gmin and the Boltzmann sum
                                                for conformer in species[structure]:
                                                    if cosmo or options.solv:
                                                        g_rel = thermo_data[conformer].solv_qhg - g_min
                                                    else:
                                                        g_rel = thermo_data[conformer].qh_gibbs_free_energy - g_min
                                                    boltz_fac = math.exp(-g_rel * J_TO_AU / GAS_CONSTANT / options.temperature)
                                                    boltz_prob = boltz_fac / boltz_sum
                                                    if hasattr(thermo_data[conformer], "sp_energy") and thermo_data[
                                                        conformer].sp_energy is not '!':
                                                        spc_abs += thermo_data[conformer].sp_energy * boltz_prob
                                                    if hasattr(thermo_data[conformer], "sp_energy") and thermo_data[conformer].sp_energy is '!':
                                                        sys.exit("\n   Not all files contain a SPC value, relative values will not be calculated.\n")
                                                    e_abs += thermo_data[conformer].scf_energy * boltz_prob
                                                    zpe_abs += thermo_data[conformer].zpe * boltz_prob
                                                    if cosmo or options.solv:
                                                        zero_conf += thermo_data[conformer].solv_qhg * boltz_prob
                                                        rel_val += thermo_data[conformer].solv_qhg * boltz_prob
                                                    else:
                                                        zero_conf += thermo_data[
                                                                         conformer].qh_gibbs_free_energy * boltz_prob
                                                        rel_val += thermo_data[
                                                                       conformer].qh_gibbs_free_energy * boltz_prob
                                                    if options.gconf:  # Default calculate gconf correction for conformers
                                                        h_conf += thermo_data[conformer].enthalpy * boltz_prob
                                                        s_conf += thermo_data[conformer].entropy * boltz_prob
                                                        s_conf += -GAS_CONSTANT / J_TO_AU * boltz_prob * math.log(boltz_prob)

                                                        qh_conf += thermo_data[conformer].qh_enthalpy * boltz_prob
                                                        qs_conf += thermo_data[conformer].qh_entropy * boltz_prob
                                                        qs_conf += -GAS_CONSTANT / J_TO_AU * boltz_prob * math.log(boltz_prob)

                                                    #else:
                                                    h_abs += thermo_data[conformer].enthalpy * boltz_prob
                                                    s_abs += thermo_data[conformer].entropy * boltz_prob
                                                    g_abs += thermo_data[conformer].gibbs_free_energy * boltz_prob

                                                    qh_abs += thermo_data[conformer].qh_enthalpy * boltz_prob
                                                    qs_abs += thermo_data[conformer].qh_entropy * boltz_prob
                                                    qhg_abs += thermo_data[
                                                                   conformer].qh_gibbs_free_energy * boltz_prob
                                                    solv_qhg_abs += thermo_data[conformer].solv_qhg * boltz_prob
                                                    if cosmo or options.solv:
                                                        self.g_qhgvals[n][i][j].append(thermo_data[conformer].solv_qhg)
                                                    else:
                                                        self.g_qhgvals[n][i][j].append(thermo_data[conformer].qh_gibbs_free_energy)
                                                if options.gconf:
                                                    h_adj = h_conf - min_conf.enthalpy
                                                    h_tot = min_conf.enthalpy + h_adj
                                                    s_adj = s_conf - min_conf.entropy
                                                    s_tot = min_conf.entropy + s_adj
                                                    g_corr = h_tot - options.temperature * s_tot
                                                    qh_adj = qh_conf - min_conf.qh_enthalpy
                                                    qh_tot = min_conf.qh_enthalpy + qh_adj
                                                    qs_adj = qs_conf - min_conf.qh_entropy
                                                    qs_tot = min_conf.qh_entropy + qs_adj
                                                    if options.QH:
                                                        qg_corr = qh_tot - options.temperature * qs_tot
                                                    else:
                                                        qg_corr = h_tot - options.temperature * qs_tot
                                                    solv_qhg_corr = solv_qhg_abs + qg_corr - h_abs + options.temperature * qs_abs
                                            self.g_species_qhgzero[n][i].append(zero_conf)  # Raw data for graphing
                                    except KeyError:
                                        log.write("   Warning! Structure " + structure + ' has not been defined correctly in ' + file + '\n')
                                        sys.exit("   Please edit " + file + " and try again\n")
                                    self.species[n].append(point)
                                    self.e_abs[n].append(e_abs)
                                    self.spc_abs[n].append(spc_abs)
                                    self.zpe_abs[n].append(zpe_abs)
                                    conformers, single_structure, mix = False, False, False
                                    self.g_rel_val[n].append(rel_val)
                                    for structure in point_structures:
                                        if not isinstance(species[structure], list):
                                            single_structure = True
                                        else:
                                            conformers = True
                                    if conformers and single_structure:
                                        mix = True

                                    if options.gconf and min_conf is not False:
                                        if mix:
                                            h_mix = h_tot + h_abs
                                            s_mix = s_tot + s_abs
                                            g_mix = g_corr + g_abs
                                            qh_mix = qh_tot + qh_abs
                                            qs_mix = qs_tot + qs_abs
                                            qg_mix = qg_corr + qhg_abs
                                            solv_qhg_mix = solv_qg_corr + solv_qhg_abs
                                            self.h_abs[n].append(h_mix)
                                            self.s_abs[n].append(s_mix)
                                            self.g_abs[n].append(g_mix)
                                            self.qh_abs[n].append(qh_mix)
                                            self.qs_abs[n].append(qs_mix)
                                            self.qhg_abs[n].append(qg_mix)
                                            self.solv_qhg_abs[n].append(solv_qhg_mix)
                                        elif conformers:
                                            self.h_abs[n].append(h_tot)
                                            self.s_abs[n].append(s_tot)
                                            self.g_abs[n].append(g_corr)
                                            self.qh_abs[n].append(qh_tot)
                                            self.qs_abs[n].append(qs_tot)
                                            self.qhg_abs[n].append(qg_corr)
                                            self.solv_qhg_abs[n].append(solv_qhg_corr)
                                    else:
                                        self.h_abs[n].append(h_abs)
                                        self.s_abs[n].append(s_abs)
                                        self.g_abs[n].append(g_abs)
                                        self.qh_abs[n].append(qh_abs)
                                        self.qs_abs[n].append(qs_abs)
                                        self.qhg_abs[n].append(qhg_abs)
                                        self.solv_qhg_abs[n].append(solv_qhg_abs)
                                else:
                                    self.species[n].append('none')
                                    self.e_abs[n].append(float('nan'))

                            n = n + 1
                        except IndexError:
                            pass

# Graph a reaction profile
def graph_reaction_profile(graph_data, options, log):

    try:
        import matplotlib.pyplot as plt
        import matplotlib.path as mpath
        import matplotlib.patches as mpatches

        log.write("\n   Graphing Reaction Profile\n")
        data = {}
        # Get PES data
        for i, path in enumerate(graph_data.path):
            g_data = []
            zero_val = graph_data.qhg_zero[i][0]
            for j, e_abs in enumerate(graph_data.e_abs[i]):
                species = graph_data.qhg_abs[i][j]
                relative = species - zero_val
                if graph_data.units == 'kJ/mol':
                    formatted_g = J_TO_AU / 1000.0 * relative
                else:
                    formatted_g = KCAL_TO_AU * relative  # Defaults to kcal/mol
                g_data.append(formatted_g)
            data[path] = g_data

        # Grab any additional formatting for graph
        with open(options.graph) as f:
            yaml = f.readlines()
        ylim, color, show_conf, show_gconf, show_title = None, None, True, False, True
        label_point, label_xaxis, dpi, dec, legend, colors, gridlines, title = True, True, False, 2, True, None, False, None
        for i, line in enumerate(yaml):
            if line.strip().find('FORMAT') > -1:
                for j, line in enumerate(yaml[i + 1:]):
                    if line.strip().find('ylim') > -1:
                        try:
                            ylim = line.strip().replace(':', '=').split("=")[1].replace(' ', '').strip().split(',')
                        except IndexError:
                            pass
                    if line.strip().find('color') > -1:
                        try:
                            colors = line.strip().replace(':', '=').split("=")[1].replace(' ', '').strip().split(',')
                        except IndexError:
                            pass
                    if line.strip().find('title') > -1:
                        try:
                            title_input = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0]
                            if title_input == 'false' or title_input == 'False':
                                show_title = False
                            else:
                                title = title_input
                        except IndexError:
                            pass
                    if line.strip().find('dec') > -1:
                        try:
                            dec = int(line.strip().replace(':', '=').split("=")[1].strip().split(',')[0])
                        except IndexError:
                            pass
                    if line.strip().find('pointlabel') > -1:
                        try:
                            label_input = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0].lower()
                            if label_input == 'false':
                                label_point = False
                        except IndexError:
                            pass
                    if line.strip().find('show_conformers') > -1:
                        try:
                            conformers = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0].lower()
                            if conformers == 'false':
                                show_conf = False
                        except IndexError:
                            pass
                    if line.strip().find('show_gconf') > -1:
                        try:
                            gconf_input = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0].lower()
                            if gconf_input == 'true':
                                show_gconf = True
                        except IndexError:
                            pass
                    if line.strip().find('xlabel') > -1:
                        try:
                            label_input = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0].lower()
                            if label_input == 'false':
                                label_xaxis = False
                        except IndexError:
                            pass
                    if line.strip().find('dpi') > -1:
                        try:
                            dpi = int(line.strip().replace(':', '=').split("=")[1].strip().split(',')[0])
                        except IndexError:
                            pass
                    if line.strip().find('legend') > -1:
                        try:
                            legend_input = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0].lower()
                            if legend_input == 'false':
                                legend = False
                        except IndexError:
                            pass
                    if line.strip().find('gridlines') > -1:
                        try:
                            gridline_input = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0].lower()
                            if gridline_input == 'true':
                                gridlines = True
                        except IndexError:
                            pass
        # Do some graphing
        Path = mpath.Path
        fig, ax = plt.subplots()
        for i, path in enumerate(graph_data.path):
            for j in range(len(data[path]) - 1):
                if colors is not None:
                    if len(colors) > 1:
                        color = colors[i]
                    else:
                        color = colors[0]
                else:
                    color = 'k'
                    colors = ['k']
                if j == 0:
                    path_patch = mpatches.PathPatch(
                        Path([(j, data[path][j]), (j + 0.5, data[path][j]), (j + 0.5, data[path][j + 1]),
                              (j + 1, data[path][j + 1])],
                             [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                        label=path, fc="none", transform=ax.transData, color=color)
                else:
                    path_patch = mpatches.PathPatch(
                        Path([(j, data[path][j]), (j + 0.5, data[path][j]), (j + 0.5, data[path][j + 1]),
                              (j + 1, data[path][j + 1])],
                             [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                        fc="none", transform=ax.transData, color=color)
                ax.add_patch(path_patch)
                plt.hlines(data[path][j], j - 0.15, j + 0.15)
            plt.hlines(data[path][-1], len(data[path]) - 1.15, len(data[path]) - 0.85)

        if show_conf:
            markers = ['o', 's', 'x', 'P', 'D']
            for i in range(len(graph_data.g_qhgvals)):  # i = reaction pathways
                for j in range(len(graph_data.g_qhgvals[i])):  # j = reaction steps
                    for k in range(len(graph_data.g_qhgvals[i][j])):  # k = species
                        zero_val = graph_data.g_species_qhgzero[i][j][k]
                        points = graph_data.g_qhgvals[i][j][k]
                        points[:] = [((x - zero_val) + (graph_data.qhg_abs[i][j] - graph_data.qhg_zero[i][0]) + (
                                graph_data.g_rel_val[i][j] - graph_data.qhg_abs[i][j])) * KCAL_TO_AU for x in points]
                        if len(colors) > 1:
                            jitter(points, colors[i], ax, j, markers[k])
                        else:
                            jitter(points, color, ax, j, markers[k])
                        if show_gconf:
                            plt.hlines((graph_data.g_rel_val[i][j] - graph_data.qhg_zero[i][0]) * KCAL_TO_AU, j - 0.15,
                                       j + 0.15, linestyles='dashed')

        # Annotate points with energy level
        if label_point:
            for i, path in enumerate(graph_data.path):
                for i, point in enumerate(data[path]):
                    if dec is 1:
                        ax.annotate("{:.1f}".format(point), (i, point - fig.get_figheight() * fig.dpi * 0.025),
                                    horizontalalignment='center')
                    else:
                        ax.annotate("{:.2f}".format(point), (i, point - fig.get_figheight() * fig.dpi * 0.025),
                                    horizontalalignment='center')
        if ylim is not None:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        if show_title:
            if title is not None:
                ax.set_title(title)
            else:
                ax.set_title("Reaction Profile")
        ax.set_ylabel(r"$G_{rel}$ (kcal / mol)")
        plt.minorticks_on()
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.tick_params(which='minor', labelright=True, right=True)
        ax.tick_params(labelright=True, right=True)
        if gridlines:
            ax.yaxis.grid(linestyle='--', linewidth=0.5)
            ax.xaxis.grid(linewidth=0)
        ax_label = []
        xaxis_text = []
        newax_text_list = []
        for i, path in enumerate(graph_data.path):
            newax_text = []
            ax_label.append(path)
            for j, e_abs in enumerate(graph_data.e_abs[i]):
                if i is 0:
                    xaxis_text.append(graph_data.species[i][j])
                else:
                    newax_text.append(graph_data.species[i][j])
            newax_text_list.append(newax_text)
        # Label rxn steps
        if label_xaxis:
            if colors is not None:
                plt.xticks(range(len(xaxis_text)), xaxis_text, color=colors[0])
            else:
                plt.xticks(range(len(xaxis_text)), xaxis_text, color='k')
            locs, labels = plt.xticks()
            newax = []
            for i in range(len(ax_label)):
                if i > 0:
                    y = ax.twiny()
                    newax.append(y)
            for i in range(len(newax)):
                newax[i].set_xticks(locs)
                newax[i].set_xlim(ax.get_xlim())
                if len(colors) > 1:
                    newax[i].tick_params(axis='x', colors=colors[i + 1])
                else:
                    newax[i].tick_params(axis='x', colors='k')
                newax[i].set_xticklabels(newax_text_list[i + 1])
                newax[i].xaxis.set_ticks_position('bottom')
                newax[i].xaxis.set_label_position('bottom')
                newax[i].xaxis.set_ticks_position('none')
                newax[i].spines['bottom'].set_position(('outward', 15 * (i + 1)))
                newax[i].spines['bottom'].set_visible(False)
        else:
            plt.xticks(range(len(xaxis_text)))
            ax.xaxis.set_ticklabels([])
        if legend:
            plt.legend()
        if dpi is not False:
            plt.savefig('Rxn_profile_' + options.graph.split('.')[0] + '.png', dpi=dpi)
        plt.show()

    except ImportError:
        log.write("\n\n   Warning! matplotlib module is not installed, reaction profile will not be graphed.")
        log.write("\n   To install matplotlib, run the following commands: \n\t   python -m pip install -U pip" +
                  "\n\t   python -m pip install -U matplotlib\n\n")

# Scatter points that may overlap when graphing
def jitter(datasets, color, ax, nx, marker, edgecol='black'):
    import numpy as np
    for i, p in enumerate(datasets):
        y = [p]
        x = np.random.normal(nx, 0.015, size=len(y))
        ax.plot(x, y, alpha=0.5, markersize=7, color=color, marker=marker, markeredgecolor=edgecol,
                markeredgewidth=1, linestyle='None')

# Read solvation free energies from a COSMO-RS dat file
def cosmo_rs_out(datfile, names, interval=False):
    gsolv = {}
    if os.path.exists(datfile):
        with open(datfile) as f:
            data = f.readlines()
    else:
        raise ValueError("File {} does not exist".format(datfile))

    temp = 0
    t_interval = []
    gsolv_dicts = []
    found = False
    oldtemp = 0
    gsolv_temp = {}
    if interval:
        for i, line in enumerate(data):
            for name in names:
                if line.find('(' + name.split('.')[0] + ')') > -1 and line.find('Compound') > -1:
                    if data[i - 5].find('Temperature') > -1:
                        temp = data[i - 5].split()[2]
                    if float(temp) > float(interval[0]) and float(temp) < float(interval[1]):
                        if float(temp) not in t_interval:
                            t_interval.append(float(temp))
                        if data[i + 10].find('Gibbs') > -1:
                            gsolv = float(data[i + 10].split()[6].strip()) / KCAL_TO_AU
                            gsolv_temp[name] = gsolv

                            found = True
            if found:
                if oldtemp is 0:
                    oldtemp = temp
                if temp is not oldtemp:
                    gsolv_dicts.append(gsolv)  # Store dict at one temp
                    gsolv = {}  # Clear gsolv
                    gsolv.update(gsolv_temp)  # Grab the first one for the new temp
                    oldtemp = temp
                gsolv.update(gsolv_temp)
                gsolv_temp = {}
                found = False
        gsolv_dicts.append(gsolv)  # Grab last dict
    else:
        for i, line in enumerate(data):
            for name in names:
                if line.find('(' + name.split('.')[0] + ')') > -1 and line.find('Compound') > -1:
                    if data[i + 11].find('Gibbs') > -1:
                        gsolv = float(data[i + 11].split()[6].strip()) / KCAL_TO_AU
                        gsolv[name] = gsolv

    if interval:
        return t_interval, gsolv_dicts
    else:
        return gsolv

# Translational energy evaluation
# Depends on temperature
def calc_translational_energy(temperature):
    """
    Calculates the translational energy (J/mol) of an ideal gas
    i.e. non-interacting molecules so molar energy = Na * atomic energy.
    This approximation applies to all energies and entropies computed within
    Etrans = 3/2 RT!
    """
    energy = 1.5 * GAS_CONSTANT * temperature
    return energy

# Rotational energy evaluation
# Depends on molecular shape and temperature
def calc_rotational_energy(zpe, symmno, temperature, linear):
    """
    Calculates the rotational energy (J/mol)
    Etrans = 0 (atomic) ; RT (linear); 3/2 RT (non-linear)
    """
    if zpe == 0.0:
        energy = 0.0
    elif linear == 1:
        energy = GAS_CONSTANT * temperature
    else:
        energy = 1.5 * GAS_CONSTANT * temperature
    return energy

# Vibrational energy evaluation
# Depends on frequencies, temperature and scaling factor: default = 1.0
def calc_vibrational_energy(frequency_wn, temperature, freq_scale_factor, fract_modelsys):
    """
    Calculates the vibrational energy contribution (J/mol).
    Includes ZPE (0K) and thermal contributions
    Evib = R * Sum(0.5 hv/k + (hv/k)/(e^(hv/KT)-1))
    """
    if fract_modelsys is not False:
        freq_scale_factor = [freq_scale_factor[0] * fract_modelsys[i] + freq_scale_factor[1] * (1.0 - fract_modelsys[i])
                             for i in range(len(fract_modelsys))]
        factor = [(PLANCK_CONSTANT * frequency_wn[i] * SPEED_OF_LIGHT * freq_scale_factor[i]) / (BOLTZMANN_CONSTANT * temperature)
                  for i in range(len(frequency_wn))]
    else:
        factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT * temperature) for freq in frequency_wn]
    # Error occurs if T is too low when performing math.exp
    for entry in factor:
        if entry > math.log(sys.float_info.max):
            sys.exit("\nx  Warning! Temperature may be too low to calculate vibrational energy. Please adjust using the `-t` option and try again.\n")

    energy = [entry * GAS_CONSTANT * temperature * (0.5 + (1.0 / (math.exp(entry) - 1.0)))
              for entry in factor]

    return sum(energy)

# Vibrational Zero point energy evaluation
# Depends on frequencies and scaling factor: default = 1.0
def calc_zeropoint_energy(frequency_wn, freq_scale_factor, fract_modelsys):
    """
    Calculates the vibrational ZPE (J/mol)
    EZPE = Sum(0.5 hv/k)
    """
    if fract_modelsys is not False:
        freq_scale_factor = [freq_scale_factor[0] * fract_modelsys[i] + freq_scale_factor[1] * (1.0 - fract_modelsys[i])
                             for i in range(len(fract_modelsys))]
        factor = [(PLANCK_CONSTANT * frequency_wn[i] * SPEED_OF_LIGHT * freq_scale_factor[i]) / (BOLTZMANN_CONSTANT)
                  for i in range(len(frequency_wn))]
    else:
        factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT)
                  for freq in frequency_wn]
    energy = [0.5 * entry * GAS_CONSTANT for entry in factor]
    return sum(energy)

# Computed the amount of accessible free space (ml per L) in solution
# accessible to a solute immersed in bulk solvent, i.e. this is the volume
# not occupied by solvent molecules, calculated using literature values for
# molarity and B3LYP/6-31G* computed molecular volumes.
def get_free_space(solv):
    """
    Calculates the free space in a litre of bulk solvent, based on
    Shakhnovich and Whitesides (J. Org. Chem. 1998, 63, 3821-3830)
    """
    solvent_list = ["none", "H2O", "toluene", "DMF", "AcOH", "chloroform"]
    molarity = [1.0, 55.6, 9.4, 12.9, 17.4, 12.5]  # mol/l
    molecular_vol = [1.0, 27.944, 149.070, 77.442, 86.10, 97.0]  # Angstrom^3

    nsolv = 0
    for i in range(0, len(solvent_list)):
        if solv == solvent_list[i]:
            nsolv = i
    solv_molarity = molarity[nsolv]
    solv_volume = molecular_vol[nsolv]
    if nsolv > 0:
        v_free = 8 * ((1E27 / (solv_molarity * AVOGADRO_CONSTANT)) ** 0.333333 - solv_volume ** 0.333333) ** 3
        freespace = v_free * solv_molarity * AVOGADRO_CONSTANT * 1E-24
    else:
        freespace = 1000.0
    return freespace

# Translational entropy evaluation
# Depends on mass, concentration, temperature, solvent free space: default = 1000.0
def calc_translational_entropy(molecular_mass, conc, temperature, solv):
    """
    Calculates the translational entropic contribution (J/(mol*K)) of an ideal gas.
    Needs the molecular mass. Convert mass in amu to kg; conc in mol/l to number per m^3
    Strans = R(Ln(2pimkT/h^2)^3/2(1/C)) + 1 + 3/2)
    """
    lmda = ((2.0 * math.pi * molecular_mass * AMU_to_KG * BOLTZMANN_CONSTANT * temperature) ** 0.5) / PLANCK_CONSTANT
    freespace = get_free_space(solv)
    ndens = conc * 1000 * AVOGADRO_CONSTANT / (freespace / 1000.0)
    entropy = GAS_CONSTANT * (2.5 + math.log(lmda ** 3 / ndens))
    return entropy

# Electronic entropy evaluation
# Depends on multiplicity
def calc_electronic_entropy(multiplicity):
    """
    Calculates the electronic entropic contribution (J/(mol*K)) of the molecule
    Selec = R(Ln(multiplicity)
    """
    entropy = GAS_CONSTANT * (math.log(multiplicity))
    return entropy

# Rotational entropy evaluation
# Depends on molecular shape and temp.
def calc_rotational_entropy(zpe, linear, symmno, rotemp, temperature):
    """
    Calculates the rotational entropy (J/(mol*K))
    Strans = 0 (atomic) ; R(Ln(q)+1) (linear); R(Ln(q)+3/2) (non-linear)
    """
    if rotemp == [0.0, 0.0, 0.0] or zpe == 0.0:  # Monatomic
        entropy = 0.0
    else:
        if len(rotemp) == 1:  # Diatomic or linear molecules
            linear = 1
            qrot = temperature / rotemp[0]
        elif len(rotemp) == 2:  # Possible gaussian problem with linear triatomic
            linear = 2
        else:
            qrot = math.pi * temperature ** 3 / (rotemp[0] * rotemp[1] * rotemp[2])
            qrot = qrot ** 0.5
        if linear == 1:
            entropy = GAS_CONSTANT * (math.log(qrot / symmno) + 1)
        elif linear == 2:
            entropy = 0.0
        else:
            entropy = GAS_CONSTANT * (math.log(qrot / symmno) + 1.5)
    return entropy

# Rigid rotor harmonic oscillator (RRHO) entropy evaluation - this is the default treatment
def calc_rrho_entropy(frequency_wn, temperature, freq_scale_factor, fract_modelsys):
    """
    Entropic contributions (J/(mol*K)) according to a rigid-rotor
    harmonic-oscillator description for a list of vibrational modes
    Sv = RSum(hv/(kT(e^(hv/kT)-1) - ln(1-e^(-hv/kT)))
    """
    if fract_modelsys is not False:
        freq_scale_factor = [freq_scale_factor[0] * fract_modelsys[i] + freq_scale_factor[1] * (1.0 - fract_modelsys[i])
                             for i in range(len(fract_modelsys))]
        factor = [(PLANCK_CONSTANT * frequency_wn[i] * SPEED_OF_LIGHT * freq_scale_factor[i]) /
                  (BOLTZMANN_CONSTANT * temperature) for i in range(len(frequency_wn))]
    else:
        factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT * temperature)
                  for freq in frequency_wn]
    entropy = [entry * GAS_CONSTANT / (math.exp(entry) - 1) - GAS_CONSTANT * math.log(1 - math.exp(-entry))
               for entry in factor]
    return entropy

# Quasi-rigid rotor harmonic oscillator energy evaluation
# used for calculating quasi-harmonic enthalpy
def calc_qRRHO_energy(frequency_wn, temperature, freq_scale_factor):
    """
    Head-Gordon RRHO-vibrational energy contribution (J/mol*K) of
    vibrational modes described by a rigid-rotor harmonic approximation
    V_RRHO = 1/2(Nhv) + RT(hv/kT)e^(-hv/kT)/(1-e^(-hv/kT))
    """
    factor = [PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor for freq in frequency_wn]
    energy = [0.5 * AVOGADRO_CONSTANT * entry + GAS_CONSTANT * temperature * entry / BOLTZMANN_CONSTANT
              / temperature * math.exp(-entry / BOLTZMANN_CONSTANT / temperature) /
              (1 - math.exp(-entry / BOLTZMANN_CONSTANT / temperature)) for entry in factor]
    return energy

# Free rotor entropy evaluation
# used for low frequencies below the cut-off if qs=grimme is specified
def calc_freerot_entropy(frequency_wn, temperature, freq_scale_factor, fract_modelsys, symmno):
    """
    Entropic contributions (J/(mol*K)) according to a free-rotor
    description for a list of vibrational modes
    Sr = R(1/2 - ln(symmno)+ 1/2ln((8pi^3u'kT/h^2))
    """
    # This is the average moment of inertia used by Grimme
    bav = 1.00e-44

    if fract_modelsys is not False:
        freq_scale_factor = [freq_scale_factor[0] * fract_modelsys[i] + freq_scale_factor[1] * (1.0 - fract_modelsys[i])
                             for i in range(len(fract_modelsys))]
        mu = [PLANCK_CONSTANT / (8 * math.pi ** 2 * frequency_wn[i] * SPEED_OF_LIGHT * freq_scale_factor[i]) for i in
              range(len(frequency_wn))]
    else:
        mu = [PLANCK_CONSTANT / (8 * math.pi ** 2 * freq * SPEED_OF_LIGHT * freq_scale_factor) for freq in frequency_wn]
    mu_primed = [entry * bav / (entry + bav) for entry in mu]
    factor = [8 * math.pi ** 3 * entry * BOLTZMANN_CONSTANT * temperature / (PLANCK_CONSTANT * symmno) ** 2 for entry in mu_primed]
    entropy = [(0.5 + math.log(entry ** 0.5)) * GAS_CONSTANT for entry in factor]
    return entropy

# A damping function to interpolate between RRHO and free rotor vibrational entropy values
def calc_damp(frequency_wn, freq_cutoff):
    alpha = 4
    damp = [1 / (1 + (freq_cutoff / entry) ** alpha) for entry in frequency_wn]
    return damp

# Calculate selectivity - enantioselectivity/diastereomeric ratio
# based on boltzmann factors of given stereoisomers
def get_selectivity(files, options, dup_list, boltz_facs, boltz_sum, log):
    # Grab files for selectivity calcs
    # list the directories to look in
    dirs = []
    for file in files:
        dirs.append(os.path.dirname(file))
    dirs = list(set(dirs))

    a_files, b_files, a_sum, b_sum, failed, pref = [], [], 0.0, 0.0, False, ''

    pattern = options.ee
    try:
        [a_regex,b_regex] = pattern.split(':')
        [a_regex,b_regex] = [a_regex.strip(), b_regex.strip()]

        A = ''.join(a for a in a_regex if a.isalnum())
        B = ''.join(b for b in b_regex if b.isalnum())

        for dir in dirs:
            a_files.extend(glob(dir+'/'+a_regex))
            b_files.extend(glob(dir+'/'+b_regex))
    except:
        pass

    if len(a_files) is 0 or len(b_files) is 0:
        log.write("\n   Warning! Filenames have not been formatted correctly for determining selectivity\n")
        log.write("   Make sure the filename contains either " + A + " or " + B + "\n")
        sys.exit("   Please edit either your filenames or selectivity pattern argument and try again\n")

    # Grab Boltzmann sums
    for file in files:
        if file not in [dup[1] for dup in dup_list]:
            for a_file in a_files:
                if file in a_file:
                    a_sum += boltz_facs[file] / boltz_sum
            for b_file in b_files:
                if file in b_file:
                    b_sum += boltz_facs[file] / boltz_sum

    # Get ratios
    A_round = round(a_sum * 100)
    B_round = round(b_sum * 100)
    r = str(A_round) + ':' + str(B_round)
    if a_sum > b_sum:
        pref = A
        try:
            ratio = a_sum / b_sum
            if ratio < 3:
                ratio = str(round(ratio, 1)) + ':1'
            else:
                ratio = str(round(ratio)) + ':1'
        except ZeroDivisionError:
            ratio = '1:0'
    else:
        pref = B
        try:
            ratio = b_sum / a_sum
            if ratio < 3:
                ratio = '1:' + str(round(ratio, 1))
            else:
                ratio = '1:' + str(round(ratio))
        except ZeroDivisionError:
            ratio = '0:1'
    ee = (a_sum - b_sum) * 100.
    if ee == 0:
        log.write("\n   Warning! No files found for selectivity analysis, adjust the names and try again.\n")
        failed = True
    ee = abs(ee)
    if ee > 99.99:
        ee = 99.99
    try:
        dd_free_energy = GAS_CONSTANT / J_TO_AU * options.temperature * math.log((50 + abs(ee) / 2.0) / (50 - abs(ee) / 2.0)) * KCAL_TO_AU
    except ZeroDivisionError:
        dd_free_energy = 0.0

    if not failed:
        selec_stars = "   " + '*' * 109
        log.write("\n   " + '{:<39} {:>13} {:>13} {:>13} {:>13} {:>13}'.format("Selectivity", "Excess (%)", "Ratio (%)", "Ratio", "Major", "DDG kcal/mol"), thermodata=True)
        log.write("\n" + selec_stars)
        log.write('\no {:<40} {:13.2f} {:>13} {:>13} {:>13} {:13.2f}'.format('', ee, r, ratio, pref,
                                                                             dd_free_energy), thermodata=True)
        log.write("\n" + selec_stars + "\n")

    return ee, r, ratio, dd_free_energy, failed, pref

# Obtain Boltzmann factors, Boltzmann sums, and weighted free energy values
# used for --ee and --boltz options
def get_boltz(files, thermo_data, options, clusters, dup_list):
    boltz_facs, weighted_free_energy, e_rel, e_min, boltz_sum = {}, {}, {}, sys.float_info.max, 0.0

    for file in files:  # Need the most stable structure
        bbe = thermo_data[file]
        if hasattr(bbe, "qh_gibbs_free_energy"):
            if bbe.qh_gibbs_free_energy != None:
                if bbe.qh_gibbs_free_energy < e_min:
                    e_min = bbe.qh_gibbs_free_energy

    if options.clustering:
        for n, cluster in enumerate(clusters):
            boltz_facs['cluster-' + alphabet[n].upper()] = 0.0
            weighted_free_energy['cluster-' + alphabet[n].upper()] = 0.0

    # Calculate E_rel and Boltzmann factors
    for file in files:
        if file not in [dup[1] for dup in dup_list]:
            bbe = thermo_data[file]
            if hasattr(bbe, "qh_gibbs_free_energy"):
                if bbe.qh_gibbs_free_energy != None:
                    e_rel[file] = bbe.qh_gibbs_free_energy - e_min
                    boltz_facs[file] = math.exp(-e_rel[file] * J_TO_AU / GAS_CONSTANT / options.temperature)
                    if options.clustering:
                        for n, cluster in enumerate(clusters):
                            for structure in cluster:
                                if structure == file:
                                    boltz_facs['cluster-' + alphabet[n].upper()] += math.exp(
                                        -e_rel[file] * J_TO_AU / GAS_CONSTANT / options.temperature)
                                    weighted_free_energy['cluster-' + alphabet[n].upper()] += math.exp(
                                        -e_rel[file] * J_TO_AU / GAS_CONSTANT / options.temperature) * bbe.qh_gibbs_free_energy
                    boltz_sum += math.exp(-e_rel[file] * J_TO_AU / GAS_CONSTANT / options.temperature)

    return boltz_facs, weighted_free_energy, boltz_sum

# Check for duplicate species from among all files based on energy, rotational constants and frequencies
# Energy cutoff = 1 microHartree; RMS Rotational Constant cutoff = 1kHz; RMS Freq cutoff = 10 wavenumbers
def check_dup(files, thermo_data, log):
    e_cutoff = 1e-4
    ro_cutoff = 1e-4
    freq_cutoff = 100
    mae_freq_cutoff = 10
    max_freq_cutoff = 10
    dup_list = []
    freq_diff, mae_freq_diff, max_freq_diff, e_diff, ro_diff = 100, 3, 10, 1, 1
    log.write('\n\no  Checking for duplicates')
    for i, file in enumerate(files):
        for j in range(0, i):
            bbe_i, bbe_j = thermo_data[files[i]], thermo_data[files[j]]
            if hasattr(bbe_i, "scf_energy") and hasattr(bbe_j, "scf_energy"):
                e_diff = bbe_i.scf_energy - bbe_j.scf_energy
            if hasattr(bbe_i, "roconst") and hasattr(bbe_j, "roconst"):
                if len(bbe_i.roconst) == len(bbe_j.roconst):
                    ro_diff = np.linalg.norm(np.array(bbe_i.roconst) - np.array(bbe_j.roconst))
            if hasattr(bbe_i, "frequency_wn") and hasattr(bbe_j, "frequency_wn"):
                if len(bbe_i.frequency_wn) == len(bbe_j.frequency_wn) and len(bbe_i.frequency_wn) > 0:
                    freq_diff = [np.linalg.norm(freqi - freqj) for freqi, freqj in
                                 zip(bbe_i.frequency_wn, bbe_j.frequency_wn)]
                    mae_freq_diff, max_freq_diff = np.mean(freq_diff), np.max(freq_diff)
                elif len(bbe_i.frequency_wn) == len(bbe_j.frequency_wn) and len(bbe_i.frequency_wn) == 0:
                    mae_freq_diff, max_freq_diff = 0., 0.
            #print(e_diff, ro_diff, mae_freq_diff, max_freq_diff)
            if e_diff < e_cutoff and ro_diff < ro_cutoff and mae_freq_diff < mae_freq_cutoff and max_freq_diff < max_freq_cutoff:
                log.write('\nx  {} is a duplicate or enantiomer of {}'.format(files[j].rsplit('.', 1)[0],
                                                                                      files[i].rsplit('.', 1)[0]))
                dup_list.append([files[i], files[j]])
    return dup_list

# Function for printing unique checks
def print_check_fails(log, check_attribute, file, attribute, option2=False):
    unique_attr = {}
    for i, attr in enumerate(check_attribute):
        if option2 is not False: attr = (attr, option2[i])
        if attr not in unique_attr:
            unique_attr[attr] = [file[i]]
        else:
            unique_attr[attr].append(file[i])
    log.write("\nx  Caution! Different {} found: ".format(attribute))
    for attr in unique_attr:
        if option2 is not False:
            if float(attr[0]) < 0:
                log.write('\n       {} {}: '.format(attr[0], attr[1]))
            else:
                log.write('\n        {} {}: '.format(attr[0], attr[1]))
        else:
            log.write('\n        -{}: '.format(attr))
        for filename in unique_attr[attr]:
            if filename is unique_attr[attr][-1]:
                log.write('{}'.format(filename))
            else:
                log.write('{}, '.format(filename))

# Perform careful checks on calculation output files
# Check for Gaussian version, solvation state/gas phase consistency, level of theory/basis set consistency,
# charge and multiplicity consistency, standard concentration used, potential linear molecule error,
# transition state verification, empirical dispersion models.
def check_files(file_data, thermo_data, options, log):
    STARS = '*' * 50
    l_o_t = ['']
    log.write("\n   Checks for thermochemistry calculations (frequency calculations):")
    log.write("\n" + STARS)
    # Check program used and version
    version_check = [file.metadata['package'] for file in file_data]
    file_check = [file.name for file in file_data]
    if all_same(version_check) != False:
        log.write("\no  Using {} in all calculations.".format(version_check[0]))
    else:
        print_check_fails(log, version_check, file_check, "programs or versions")

    # Check level of theory
    if all_same(l_o_t) is not False:
        log.write("\no  Using {} in all calculations.".format(l_o_t[0]))
    elif all_same(l_o_t) is False:
        print_check_fails(log, l_o_t, file_check, "levels of theory")

    # Check for solvent models
    solvent_check = [thermo_data[key].solvation_model[0] for key in thermo_data]
    if all_same(solvent_check):
        solvent_check = [thermo_data[key].solvation_model[1] for key in thermo_data]
        log.write("\no  Using {} in all calculations.".format(solvent_check[0]))
    else:
        solvent_check = [thermo_data[key].solvation_model[1] for key in thermo_data]
        print_check_fails(log, solvent_check, file_check, "solvation models")

    # Check for -c 1 when solvent is added
    if all_same(solvent_check):
        if solvent_check[0] == "gas phase" and str(round(options.conc, 4)) == str(round(0.0408740470708, 4)):
            log.write("\no  Using a standard concentration of 1 atm for gas phase.")
        elif solvent_check[0] == "gas phase" and str(round(options.conc, 4)) != str(round(0.0408740470708, 4)):
            log.write("\nx  Caution! Standard concentration is not 1 atm for gas phase (using {} M).".format(options.conc))
        elif solvent_check[0] != "gas phase" and str(round(options.conc, 4)) == str(round(0.0408740470708, 4)):
            log.write("\nx  Using a standard concentration of 1 atm for solvent phase (option -c 1 should be included for 1 M).")
        elif solvent_check[0] != "gas phase" and str(options.conc) == str(1.0):
            log.write("\no  Using a standard concentration of 1 M for solvent phase.")
        elif solvent_check[0] != "gas phase" and str(round(options.conc, 4)) != str(round(0.0408740470708, 4)) and str(
                options.conc) != str(1.0):
            log.write("\nx  Caution! Standard concentration is not 1 M for solvent phase (using {} M).".format(options.conc))
    if all_same(solvent_check) == False and "gas phase" in solvent_check:
        log.write("\nx  Caution! The right standard concentration cannot be determined because the calculations use a combination of gas and solvent phases.")
    if all_same(solvent_check) == False and "gas phase" not in solvent_check:
        log.write("\nx  Caution! Different solvents used, fix this issue and use option -c 1 for a standard concentration of 1 M.")

    # Check charge and multiplicity
    charge_check = [thermo_data[key].charge for key in thermo_data]
    multiplicity_check = [thermo_data[key].multiplicity for key in thermo_data]
    if all_same(charge_check) != False and all_same(multiplicity_check) != False:
        log.write("\no  Using charge {} and multiplicity {} in all calculations.".format(charge_check[0],
                                                                                         multiplicity_check[0]))
    else:
        print_check_fails(log, charge_check, file_check, "charge and multiplicity", multiplicity_check)

    # Check for duplicate structures
    dup_list = check_dup(files, thermo_data)
    if len(dup_list) == 0:
        log.write("\no  No duplicates or enantiomers found")
    else:
        log.write("\nx  Caution! Possible duplicates or enantiomers found:")
        for dup in dup_list:
            log.write('\n        {} and {}'.format(dup[0], dup[1]))

    # Check for linear molecules with incorrect number of vibrational modes
    linear_fails, linear_fails_atom, linear_fails_cart, linear_fails_files, linear_fails_list = [], [], [], [], []
    frequency_list = []

    for file in files:
        linear_fails = getoutData(file)
        linear_fails_cart.append(linear_fails.cartesians)
        linear_fails_atom.append(linear_fails.atom_types)
        linear_fails_files.append(file)
        frequency_list.append(thermo_data[file].frequency_wn)

    linear_fails_list.append(linear_fails_atom)
    linear_fails_list.append(linear_fails_cart)
    linear_fails_list.append(frequency_list)
    linear_fails_list.append(linear_fails_files)

    linear_mol_correct, linear_mol_wrong = [], []
    for i in range(len(linear_fails_list[0])):
        count_linear = 0
        if len(linear_fails_list[0][i]) == 2:
            if len(linear_fails_list[2][i]) == 1:
                linear_mol_correct.append(linear_fails_list[3][i])
            else:
                linear_mol_wrong.append(linear_fails_list[3][i])
        if len(linear_fails_list[0][i]) == 3:
            if linear_fails_list[0][i] == ['I', 'I', 'I'] or linear_fails_list[0][i] == ['O', 'O', 'O'] or \
                    linear_fails_list[0][i] == ['N', 'N', 'N'] or linear_fails_list[0][i] == ['H', 'C', 'N'] or \
                    linear_fails_list[0][i] == ['H', 'N', 'C'] or linear_fails_list[0][i] == ['C', 'H', 'N'] or \
                    linear_fails_list[0][i] == ['C', 'N', 'H'] or linear_fails_list[0][i] == ['N', 'H', 'C'] or \
                    linear_fails_list[0][i] == ['N', 'C', 'H']:
                if len(linear_fails_list[2][i]) == 4:
                    linear_mol_correct.append(linear_fails_list[3][i])
                else:
                    linear_mol_wrong.append(linear_fails_list[3][i])
            else:
                for j in range(len(linear_fails_list[0][i])):
                    for k in range(len(linear_fails_list[0][i])):
                        if k > j:
                            for l in range(len(linear_fails_list[1][i][j])):
                                if linear_fails_list[0][i][j] == linear_fails_list[0][i][k]:
                                    if linear_fails_list[1][i][j][l] > (-linear_fails_list[1][i][k][l] - 0.1) and \
                                            linear_fails_list[1][i][j][l] < (-linear_fails_list[1][i][k][l] + 0.1):
                                        count_linear = count_linear + 1
                                        if count_linear == 3:
                                            if len(linear_fails_list[2][i]) == 4:
                                                linear_mol_correct.append(linear_fails_list[3][i])
                                            else:
                                                linear_mol_wrong.append(linear_fails_list[3][i])
        if len(linear_fails_list[0][i]) == 4:
            if linear_fails_list[0][i] == ['C', 'C', 'H', 'H'] or linear_fails_list[0][i] == ['C', 'H', 'C', 'H'] or \
                    linear_fails_list[0][i] == ['C', 'H', 'H', 'C'] or linear_fails_list[0][i] == ['H', 'C', 'C', 'H'] or \
                    linear_fails_list[0][i] == ['H', 'C', 'H', 'C'] or linear_fails_list[0][i] == ['H', 'H', 'C', 'C']:
                if len(linear_fails_list[2][i]) == 7:
                    linear_mol_correct.append(linear_fails_list[3][i])
                else:
                    linear_mol_wrong.append(linear_fails_list[3][i])
    linear_correct_print, linear_wrong_print = "", ""
    for i in range(len(linear_mol_correct)):
        linear_correct_print += ', ' + linear_mol_correct[i]
    for i in range(len(linear_mol_wrong)):
        linear_wrong_print += ', ' + linear_mol_wrong[i]
    linear_correct_print = linear_correct_print[1:]
    linear_wrong_print = linear_wrong_print[1:]
    if len(linear_mol_correct) == 0:
        if len(linear_mol_wrong) == 0:
            log.write("\n-  No linear molecules found.")
        if len(linear_mol_wrong) >= 1:
            log.write("\nx  Caution! Potential linear molecules with wrong number of frequencies found "
                      "(correct number = 3N-5) -{}.".format(linear_wrong_print))
    elif len(linear_mol_correct) >= 1:
        if len(linear_mol_wrong) == 0:
            log.write("\no  All the linear molecules have the correct number of frequencies -{}.".format(linear_correct_print))
        if len(linear_mol_wrong) >= 1:
            log.write("\nx  Caution! Potential linear molecules with wrong number of frequencies found -{}. Correct "
                      "number of frequencies (3N-5) found in other calculations -{}.".format(linear_wrong_print,
                                                                                             linear_correct_print))

    # Checks whether any TS have > 1 imaginary frequency and any GS have any imaginary frequencies
    for file in files:
        bbe = thermo_data[file]
        if bbe.job_type.find('TS') > -1 and len(bbe.im_frequency_wn) != 1:
            log.write("\nx  Caution! TS {} does not have 1 imaginary frequency greater than -50 wavenumbers.".format(file))
        if bbe.job_type.find('GS') > -1 and bbe.job_type.find('TS') == -1 and len(bbe.im_frequency_wn) != 0:
            log.write("\nx  Caution: GS {} has 1 or more imaginary frequencies greater than -50 wavenumbers.".format(file))

    # Check for empirical dispersion
    dispersion_check = [thermo_data[key].empirical_dispersion for key in thermo_data]
    if all_same(dispersion_check):
        if dispersion_check[0] == 'No empirical dispersion detected':
            log.write("\n-  No empirical dispersion detected in any of the calculations.")
        else:
            log.write("\no  Using " + dispersion_check[0] + " in all calculations.")
    else:
        print_check_fails(log, dispersion_check, file_check, "dispersion models")
    log.write("\n" + STARS + "\n")

    # Check for single-point corrections
    if options.spc is not False:
        log.write("\n   Checks for single-point corrections:")
        log.write("\n" + STARS)
        names_spc, version_check_spc = [], []
        for file in files:
            name, ext = os.path.splitext(file)
            if os.path.exists(name + '_' + options.spc + '.log'):
                names_spc.append(name + '_' + options.spc + '.log')
            elif os.path.exists(name + '_' + options.spc + '.out'):
                names_spc.append(name + '_' + options.spc + '.out')

        # Check SPC program versions
        version_check_spc = [thermo_data[key].sp_version_program for key in thermo_data]
        if all_same(version_check_spc):
            log.write("\no  Using {} in all the single-point corrections.".format(version_check_spc[0]))
        else:
            print_check_fails(log, version_check_spc, file_check, "programs or versions")

        # Check SPC solvation
        solvent_check_spc = [thermo_data[key].sp_solvation_model for key in thermo_data]
        if all_same(solvent_check_spc):
            log.write("\no  Using " + solvent_check_spc[0] + " in all the single-point corrections.")
        else:
            print_check_fails(log, solvent_check_spc, file_check, "solvation models")

        # Check SPC level of theory
        l_o_t_spc = [level_of_theory(name) for name in names_spc]
        if all_same(l_o_t_spc):
            log.write("\no  Using {} in all the single-point corrections.".format(l_o_t_spc[0]))
        else:
            print_check_fails(log, l_o_t_spc, file_check, "levels of theory")

        # Check SPC charge and multiplicity
        charge_spc_check = [thermo_data[key].sp_charge for key in thermo_data]
        multiplicity_spc_check = [thermo_data[key].sp_multiplicity for key in thermo_data]
        if all_same(charge_spc_check) != False and all_same(multiplicity_spc_check) != False:
            log.write("\no  Using charge and multiplicity {} {} in all the single-point corrections.".format(
                charge_spc_check[0], multiplicity_spc_check[0]))
        else:
            print_check_fails(log, charge_spc_check, file_check, "charge and multiplicity", multiplicity_spc_check)

        # Check if the geometries of freq calculations match their corresponding structures in single-point calculations
        geom_duplic_list, geom_duplic_list_spc, geom_duplic_cart, geom_duplic_files, geom_duplic_cart_spc, geom_duplic_files_spc = [], [], [], [], [], []
        for file in files:
            geom_duplic = getoutData(file)
            geom_duplic_cart.append(geom_duplic.cartesians)
            geom_duplic_files.append(file)
        geom_duplic_list.append(geom_duplic_cart)
        geom_duplic_list.append(geom_duplic_files)

        for name in names_spc:
            geom_duplic_spc = getoutData(name)
            geom_duplic_cart_spc.append(geom_duplic_spc.cartesians)
            geom_duplic_files_spc.append(name)
        geom_duplic_list_spc.append(geom_duplic_cart_spc)
        geom_duplic_list_spc.append(geom_duplic_files_spc)
        spc_mismatching = "Caution! Potential differences found between frequency and single-point geometries -"
        if len(geom_duplic_list[0]) == len(geom_duplic_list_spc[0]):
            for i in range(len(files)):
                count = 1
                for j in range(len(geom_duplic_list[0][i])):
                    if count == 1:
                        if geom_duplic_list[0][i][j] == geom_duplic_list_spc[0][i][j]:
                            count = count
                        elif '{0:.3f}'.format(geom_duplic_list[0][i][j][0]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][0] * (-1)) or '{0:.3f}'.format(geom_duplic_list[0][i][j][0]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][0]):
                            if '{0:.3f}'.format(geom_duplic_list[0][i][j][1]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][1] * (-1)) or '{0:.3f}'.format(geom_duplic_list[0][i][j][1]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][1] * (-1)):
                                count = count
                            if '{0:.3f}'.format(geom_duplic_list[0][i][j][2]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][2] * (-1)) or '{0:.3f}'.format(
                                geom_duplic_list[0][i][j][2]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][2] * (-1)):
                                count = count
                        else:
                            spc_mismatching += ", " + geom_duplic_list[1][i]
                            count = count + 1
            if spc_mismatching == "Caution! Potential differences found between frequency and single-point geometries -":
                log.write("\no  No potential differences found between frequency and single-point geometries (based on input coordinates).")
            else:
                spc_mismatching_1 = spc_mismatching[:84]
                spc_mismatching_2 = spc_mismatching[85:]
                log.write("\nx  " + spc_mismatching_1 + spc_mismatching_2 + '.')
        else:
            log.write("\nx  One or more geometries from single-point corrections are missing.")

        # Check for SPC dispersion models
        dispersion_check_spc = [thermo_data[key].sp_empirical_dispersion for key in thermo_data]
        if all_same(dispersion_check_spc):
            if dispersion_check_spc[0] == 'No empirical dispersion detected':
                log.write("\n-  No empirical dispersion detected in any of the calculations.")
            else:
                log.write("\no  Using " + dispersion_check_spc[0] + " in all the singe-point calculations.")
        else:
            print_check_fails(log, dispersion_check_spc, file_check, "dispersion models")
        log.write("\n" + STARS + "\n")

def print_intro(options, log):
    log.write("\n\n   GoodVibes v" + __version__ + " " + options.start + "\n   " + goodvibes_ref + "\n")

    # Summary of the quasi-harmonic treatment; print out the relevant reference
    if options.temperature_interval is False:
        log.write("   Temperature = " + str(options.temperature) + " Kelvin")
    # If not at standard temp, need to correct the molarity of 1 atmosphere (assuming pressure is still 1 atm)
    if options.gas_phase:
        log.write("   Pressure = 1 atm")
    else:
        log.write("   Concentration = " + str(options.conc) + " mol/L")

    log.write('\n   All energetic values below shown in Hartree unless otherwise specified.')

    log.write("\n\no  Entropic quasi-harmonic treatment: frequency cut-off value of " + str(
        options.S_freq_cutoff) + " wavenumbers will be applied.")
    if options.QS == "grimme":
        log.write("\n   QS = Grimme: Using a mixture of RRHO and Free-rotor vibrational entropies.")
        qs_ref = grimme_ref
    elif options.QS == "truhlar":
        log.write("\n   QS = Truhlar: Using an RRHO treatment where low frequencies are adjusted to the cut-off value.")
        qs_ref = truhlar_ref
    else:
        log.fatal("\n   FATAL ERROR: Unknown quasi-harmonic model " + options.QS + " specified (QS must = grimme or truhlar).")
    log.write("\n   " + qs_ref + '\n')

    # Check if qh-H correction should be applied
    if options.QH:
        log.write("\n\n   Enthalpy quasi-harmonic treatment: frequency cut-off value of " + str(
            options.H_freq_cutoff) + " wavenumbers will be applied.")
        log.write("\n   QH = Head-Gordon: Using an RRHO treatement with an approximation term for vibrational energy.")
        qh_ref = head_gordon_ref
        log.write("\n   REF: " + qh_ref + '\n')

    # Check if D3 corrections should be applied
    if options.D3:
        log.write("\no  D3-Dispersion energy with zero-damping will be calculated and included in the energy and enthalpy terms.")
        log.write("\n   " + d3_ref + '\n')
    if options.D3BJ:
        log.write("\no  D3-Dispersion energy with Becke-Johnson damping will be calculated and added to the energy terms.")
        log.write("\n   " + d3bj_ref + '\n')
    if options.ATM:
        log.write("\n   The repulsive Axilrod-Teller-Muto 3-body term will be included in the dispersion correction.")
        log.write("\n   " + atm_ref + '\n')

    # Check if entropy symmetry correction should be applied
    if options.ssymm:
        log.write('\n   Ssymm requested. Symmetry contribution to entropy to be calculated using S. Patchkovskii\'s \n   open source software "Brute Force Symmetry Analyzer" available under GNU General Public License.')
        log.write('\n   REF: (C) 1996, 2003 S. Patchkovskii, Serguei.Patchkovskii@sympatico.ca')
        log.write('\n\n   Atomic radii used to calculate internal symmetry based on Cambridge Structural Database covalent radii.')
        log.write("\n   REF: " + csd_ref + '\n')

    # Whether linked single-point energies are to be used
    if options.spc:
        log.write("\n   Link job: combining final single point energy with thermal corrections.")

    log.write('\n'+options.command)

def print_main(files, thermo_data, options, dup_list, clusters, log):
    ''' print table of absolute values'''
    if options.QH:
        stars = "   " + "*" * 142
    else:
        stars = "   " + "*" * 128
    if options.spc is not False: stars += '*' * 14
    if options.cosmo is not False: stars += '*' * 30
    if options.imag_freq is True: stars += '*' * 9
    if options.boltz is True: stars += '*' * 7
    if options.ssymm is True: stars += '*' * 13

    # Boltzmann factors and averaging over clusters
    if options.boltz != False or options.ee != False:
        boltz_facs, weighted_free_energy, boltz_sum = get_boltz(files, thermo_data, options, clusters, dup_list)

    # Standard mode: tabulate thermochemistry ouput from file(s) at a single temperature and concentration
    if options.spc is False:
        log.write("\n\n   ")
        if options.QH:
            log.write('{:<39} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                      '{:>13}'.format("Structure", "E", "ZPE", "H", "qh-H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),
                      thermodata=True)
        else:
            log.write('{:<39} {:>13} {:>10} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "E", "ZPE", "H",
                                                                                       "T.S", "T.qh-S", "G(T)",
                                                                                       "qh-G(T)"), thermodata=True)
    else:
        log.write("\n\n   ")
        if options.QH:
            log.write('{:<39} {:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                      '{:>13}'.format("Structure", "E_SPC", "E", "ZPE", "H_SPC", "qh-H_SPC", "T.S", "T.qh-S",
                                      "G(T)_SPC", "qh-G(T)_SPC"), thermodata=True)
        else:
            log.write('{:<39} {:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                      '{:>13}'.format("Structure", "E_SPC", "E", "ZPE", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC",
                                      "qh-G(T)_SPC"), thermodata=True)
    if options.cosmo is not False:
        log.write('{:>13} {:>16}'.format("COSMO-RS", "Solv-qh-G(T)"), thermodata=True)
    if options.boltz is True:
        log.write('{:>7}'.format("Boltz"), thermodata=True)
    if options.imag_freq is True:
        log.write('{:>9}'.format("im freq"), thermodata=True)
    if options.ssymm:
        log.write('{:>13}'.format("Point Group"), thermodata=True)
    log.write("\n" + stars + "")

    for file in files:  # Loop over the output files and compute thermochemistry
        if file not in [dup[1] for dup in dup_list]:
            bbe = thermo_data[file]

            # Check for possible error in Gaussian calculation of linear molecules which can return 2 rotational constants instead of 3
            if bbe.linear_warning:
                log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                log.write('          ----   Caution! Potential invalid calculation of linear molecule from Gaussian')
            else:
                if hasattr(bbe, "gibbs_free_energy"):
                    if options.spc is not False:
                        if bbe.sp_energy != '!':
                            log.write("\no  ")
                            log.write('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]), thermodata=True)
                            log.write(' {:13.6f}'.format(bbe.sp_energy), thermodata=True)
                        if bbe.sp_energy == '!':
                            log.write("\nx  ")
                            log.write('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]), thermodata=True)
                            log.write(' {:>13}'.format('----'), thermodata=True)
                    else:
                        log.write("\no  ")
                        log.write('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]), thermodata=True)
                # Gaussian SPC file handling
                if hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                    log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                # ORCA spc files
                elif not hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                    log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                if hasattr(bbe, "scf_energy"):
                    log.write(' {:13.6f}'.format(bbe.scf_energy), thermodata=True)
                # No freqs found
                if not hasattr(bbe, "gibbs_free_energy"):
                    log.write("   Warning! Couldn't find frequency information ...")
                else:
                    if not options.media:
                        if all(getattr(bbe, attrib) for attrib in
                               ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                            if options.QH:
                                log.write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(
                                    bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy, (options.temperature * bbe.entropy),
                                    (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy,
                                    bbe.qh_gibbs_free_energy), thermodata=True)
                            else:
                                log.write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                          '{:13.6f}'.format(bbe.zpe, bbe.enthalpy,
                                                            (options.temperature * bbe.entropy),
                                                            (options.temperature * bbe.qh_entropy),
                                                            bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                          thermodata=True)
                    else:
                        try:
                            from .media import solvents
                        except:
                            from media import solvents
                            # Media correction based on standard concentration of solvent
                        if options.media.lower() in solvents and options.media.lower() == \
                                os.path.splitext(os.path.basename(file))[0].lower():
                            mw_solvent = solvents[options.media.lower()][0]
                            density_solvent = solvents[options.media.lower()][1]
                            concentration_solvent = (density_solvent * 1000) / mw_solvent
                            media_correction = -(GAS_CONSTANT / J_TO_AU) * math.log(concentration_solvent)
                            if all(getattr(bbe, attrib) for attrib in ["enthalpy", "entropy", "qh_entropy",
                                                                       "gibbs_free_energy", "qh_gibbs_free_energy"]):
                                if options.QH:
                                    log.write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                              '{:13.6f}'.format(bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy,
                                                                (options.temperature * (bbe.entropy + media_correction)),
                                                                (options.temperature * (bbe.qh_entropy + media_correction)),
                                                                bbe.gibbs_free_energy + (options.temperature * (-media_correction)),
                                                                bbe.qh_gibbs_free_energy + (options.temperature * (-media_correction))),
                                              thermodata=True)
                                    log.write("  Solvent")
                                else:
                                    log.write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                              '{:13.6f}'.format(bbe.zpe, bbe.enthalpy,
                                                                (options.temperature * (bbe.entropy + media_correction)),
                                                                (options.temperature * (bbe.qh_entropy + media_correction)),
                                                                bbe.gibbs_free_energy + (options.temperature * (-media_correction)),
                                                                bbe.qh_gibbs_free_energy + (options.temperature * (-media_correction))), thermodata=True)
                                    log.write("  Solvent")
                        else:
                            if all(getattr(bbe, attrib) for attrib in ["enthalpy", "entropy", "qh_entropy",
                                                                       "gibbs_free_energy", "qh_gibbs_free_energy"]):
                                if options.QH:
                                    log.write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                              '{:13.6f}'.format(bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy, (options.temperature * bbe.entropy), (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy), thermodata=True)
                                else:
                                    log.write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                              '{:13.6f}'.format(bbe.zpe, bbe.enthalpy,
                                                                (options.temperature * bbe.entropy),
                                                                (options.temperature * bbe.qh_entropy),
                                                                bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                              thermodata=True)
            # Append requested options to end of output
            if options.cosmo and cosmo_solv is not None:
                log.write('{:13.6f} {:16.6f}'.format(cosmo_solv[file], bbe.qh_gibbs_free_energy + cosmo_solv[file]))
            if options.boltz is True:
                log.write('{:7.3f}'.format(boltz_facs[file] / boltz_sum), thermodata=True)
            if options.imag_freq is True and hasattr(bbe, "im_frequency_wn"):
                for freq in bbe.im_frequency_wn:
                    log.write('{:9.2f}'.format(freq), thermodata=True)
            if options.ssymm:
                if hasattr(bbe, "qh_gibbs_free_energy"):
                    log.write('{:>13}'.format(bbe.point_group))
                else:
                    log.write('{:>37}'.format('---'))


        # Cluster files if requested
        if options.clustering:
            dashes = "-" * (len(stars) - 3)
            for n, cluster in enumerate(clusters):
                for id, structure in enumerate(cluster):
                    if structure == file:
                        if id == len(cluster) - 1:
                            log.write("\n   " + dashes)
                            log.write("\n   " + '{name:<{var_width}} {gval:13.6f} {weight:6.2f}'.format(
                                name='Boltzmann-weighted Cluster ' + alphabet[n].upper(), var_width=len(stars) - 24,
                                gval=weighted_free_energy['cluster-' + alphabet[n].upper()] / boltz_facs[
                                    'cluster-' + alphabet[n].upper()],
                                weight=100 * boltz_facs['cluster-' + alphabet[n].upper()] / boltz_sum),
                                      thermodata=True)
                            log.write("\n   " + dashes)

    log.write("\n" + stars + "\n")

def tabulate(thermo_data, options, log):
    ''' Tabulate relative values'''
    stars = "   " + "*" * 128
    if options.spc: stars = stars + "*" * 23
    if options.gconf:
        log.write('\n   Gconf correction requested to be applied to below relative values using quasi-harmonic Boltzmann factors\n')
    for key in thermo_data:
        if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
            pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
            sys.exit(pes_error)
        if not hasattr(thermo_data[key], "sp_energy") and options.spc is not False:
            pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
            sys.exit(pes_error)
    # Interval applied to PES
    if options.temperature_interval:
        stars = stars + '*' * 22
        for i in range(len(interval)):
            bbe_vals = []
            for j in range(len(interval_bbe_data)):
                bbe_vals.append(interval_bbe_data[j][i])
            interval_thermo_data.append(dict(zip(file_list, bbe_vals)))
        j = 0
        for i in interval:
            temp = float(i)
            if options.cosmo_int is False:
                pes = get_pes(options.pes, interval_thermo_data[j], options, log, temp, options.gconf, options.QH)
            else:
                pes = get_pes(options.pes, interval_thermo_data[j], options, log, temp, options.gconf, options.QH,
                              cosmo=True)
            for k, path in enumerate(pes.path):
                if options.QH:
                    zero_vals = [pes.spc_zero[k][0], pes.e_zero[k][0], pes.zpe_zero[k][0], pes.h_zero[k][0],
                                 pes.qh_zero[k][0], temp * pes.ts_zero[k][0], temp * pes.qhts_zero[k][0],
                                 pes.g_zero[k][0], pes.qhg_zero[k][0]]
                else:
                    zero_vals = [pes.spc_zero[k][0], pes.e_zero[k][0], pes.zpe_zero[k][0], pes.h_zero[k][0],
                                 temp * pes.ts_zero[k][0], temp * pes.qhts_zero[k][0], pes.g_zero[k][0],
                                 pes.qhg_zero[k][0]]
                if options.cosmo_int:
                    zero_vals.append(pes.solv_qhg_abs[k][0])
                if pes.boltz:
                    e_sum, h_sum, g_sum, qhg_sum = 0.0, 0.0, 0.0, 0.0
                    sels = []
                    for l, e_abs in enumerate(pes.e_abs[k]):
                        if options.QH:
                            species = [pes.spc_abs[k][l], pes.e_abs[k][l], pes.zpe_abs[k][l], pes.h_abs[k][l],
                                       pes.qh_abs[k][l], temp * pes.s_abs[k][l], temp * pes.qs_abs[k][l],
                                       pes.g_abs[k][l], pes.qhg_abs[k][l]]
                        else:
                            species = [pes.spc_abs[k][l], pes.e_abs[k][l], pes.zpe_abs[k][l], pes.h_abs[k][l],
                                       temp * pes.s_abs[k][l], temp * pes.qs_abs[k][l], pes.g_abs[k][l],
                                       pes.qhg_abs[k][l]]
                        relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                        e_sum += math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / temp)
                        h_sum += math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / temp)
                        g_sum += math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / temp)
                        qhg_sum += math.exp(-relative[8] * J_TO_AU / GAS_CONSTANT / temp)
                if options.spc is False:
                    log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ")  at T: " + str(temp)))
                    if options.QH and options.cosmo_int:
                        log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)",
                                                  "qh-DG(T)", 'Solv-qh-G(T)'), thermodata=True)
                    elif options.QH:
                        log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)",
                                                  "qh-DG(T)"), thermodata=True)
                    elif options.cosmo_int:
                        log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)",
                                                  'Solv-qh-G(T)'), thermodata=True)
                    else:
                        log.write('{:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                                  thermodata=True)
                else:
                    log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ")  at T: " +
                                                        str(temp)))
                    if options.QH and options.cosmo_int:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} {:>14} {:>14}'.format(
                            " DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS", "T.qh-DS", "DG(T)_SPC",
                            "qh-DG(T)_SPC", 'Solv-qh-G(T)_SPC'), thermodata=True)
                    elif options.QH:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                                  '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS",
                                                  "T.qh-DS", "DG(T)_SPC", "qh-DG(T)_SPC"), thermodata=True)
                    elif options.cosmo_int:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                                  '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS",
                                                  "DG(T)_SPC", "qh-DG(T)_SPC", 'Solv-qh-G(T)_SPC'),
                                  thermodata=True)
                    else:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>14} '
                                  '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS",
                                                  "DG(T)_SPC", "qh-DG(T)_SPC"), thermodata=True)
                log.write("\n" + stars)

                for l, e_abs in enumerate(pes.e_abs[k]):
                    if options.QH:
                        species = [pes.spc_abs[k][l], pes.e_abs[k][l], pes.zpe_abs[k][l], pes.h_abs[k][l],
                                   pes.qh_abs[k][l], temp * pes.s_abs[k][l], temp * pes.qs_abs[k][l],
                                   pes.g_abs[k][l], pes.qhg_abs[k][l]]
                    else:
                        species = [pes.spc_abs[k][l], pes.e_abs[k][l], pes.zpe_abs[k][l], pes.h_abs[k][l],
                                   temp * pes.s_abs[k][l], temp * pes.qs_abs[k][l], pes.g_abs[k][l],
                                   pes.qhg_abs[k][l]]
                    if options.cosmo_int:
                        species.append(pes.solv_qhg_abs[k][l])
                    relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                    if pes.units == 'kJ/mol':
                        formatted_list = [J_TO_AU / 1000.0 * x for x in relative]
                    else:
                        formatted_list = [KCAL_TO_AU * x for x in relative]  # Defaults to kcal/mol
                    log.write("\no  ")
                    if options.spc is False:
                        formatted_list = formatted_list[1:]
                        format_1 = '{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} ' \
                                   '{:13.1f} {:13.1f}'
                        format_2 = '{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} ' \
                                   '{:13.2f} {:13.2f}'
                        if options.QH and options.cosmo_int:
                            if pes.dec == 1:
                                log.write(format_1.format(pes.species[k][l], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write(format_2.format(pes.species[k][l], *formatted_list), thermodata=True)
                        elif options.QH or options.cosmo_int:
                            if pes.dec == 1:
                                log.write(format_1.format(pes.species[k][l], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write(format_2.format(pes.species[k][l], *formatted_list), thermodata=True)
                        else:
                            if pes.dec == 1:
                                log.write(format_1.format(pes.species[k][l], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write(format_2.format(pes.species[k][l], *formatted_list), thermodata=True)
                    else:
                        if options.QH and options.cosmo_int:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} '
                                          '{:13.1f} {:13.1f} {:13.1f}'.format(pes.species[k][l], *formatted_list),
                                          thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f} {:13.2f}'.format(
                                        pes.species[k][l], *formatted_list), thermodata=True)
                        elif options.QH or options.cosmo_int:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(
                                        pes.species[k][l], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(
                                        pes.species[k][l], *formatted_list), thermodata=True)
                        else:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(
                                        pes.species[k][l], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.2f} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(
                                        pes.species[k][l], *formatted_list), thermodata=True)
                    if pes.boltz:
                        boltz = [math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / options.temperature) / e_sum,
                                 math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / options.temperature) / h_sum,
                                 math.exp(-relative[6] * J_TO_AU / GAS_CONSTANT / options.temperature) / g_sum,
                                 math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / options.temperature) / qhg_sum]
                        selectivity = [boltz[x] * 100.0 for x in range(len(boltz))]
                        log.write("\n  " + '{:<39} {:13.2f}%{:24.2f}%{:35.2f}%{:13.2f}%'.format('', *selectivity))
                        sels.append(selectivity)
                    formatted_list = [round(formatted_list[x], 6) for x in range(len(formatted_list))]
                if pes.boltz == 'ee' and len(sels) == 2:
                    ee = [sels[0][x] - sels[1][x] for x in range(len(sels[0]))]
                    if options.spc is False:
                        log.write("\n" + stars + "\n   " + '{:<39} {:13.1f}%{:24.1f}%{:35.1f}%{:13.1f}%'.format('ee (%)',
                                                                                                          *ee))
                    else:
                        log.write("\n" + stars + "\n   " + '{:<39} {:27.1f} {:24.1f} {:35.1f} {:13.1f} '.format('ee (%)',
                                                                                                          *ee))
                log.write("\n" + stars + "\n")
            j += 1
    else:
        if options.cosmo:
            pes = get_pes(thermo_data, options, log, cosmo=True)
        else:
            pes = get_pes(thermo_data, options, log)

        # Output the relative energy data
        for i, path in enumerate(pes.path):
            if options.QH:
                zero_vals = [pes.spc_zero[i][0], pes.e_zero[i][0], pes.zpe_zero[i][0], pes.h_zero[i][0],
                             pes.qh_zero[i][0], options.temperature * pes.ts_zero[i][0],
                             options.temperature * pes.qhts_zero[i][0], pes.g_zero[i][0], pes.qhg_zero[i][0]]
            else:
                zero_vals = [pes.spc_zero[i][0], pes.e_zero[i][0], pes.zpe_zero[i][0], pes.h_zero[i][0],
                             options.temperature * pes.ts_zero[i][0], options.temperature * pes.qhts_zero[i][0],
                             pes.g_zero[i][0], pes.qhg_zero[i][0]]
            if options.cosmo or options.solv:
                zero_vals.append(pes.solv_qhg_zero[i][0])
            if pes.boltz:
                e_sum, h_sum, g_sum, qhg_sum, solv_qhg_sum = 0.0, 0.0, 0.0, 0.0, 0.0
                sels = []
                for j, e_abs in enumerate(pes.e_abs[i]):
                    if options.QH:
                        species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                                   pes.qh_abs[i][j], options.temperature * pes.s_abs[i][j],
                                   options.temperature * pes.qs_abs[i][j], pes.g_abs[i][j], pes.qhg_abs[i][j]]
                    else:
                        species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                                   options.temperature * pes.s_abs[i][j], options.temperature * pes.qs_abs[i][j],
                                   pes.g_abs[i][j], pes.qhg_abs[i][j]]
                    if options.cosmo or options.solv:
                        species.append(pes.solv_qhg_abs[i][j])

                    relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                    e_sum += math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / options.temperature)
                    h_sum += math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / options.temperature)
                    g_sum += math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / options.temperature)
                    qhg_sum += math.exp(-relative[8] * J_TO_AU / GAS_CONSTANT / options.temperature)
                    solv_qhg_sum += math.exp(-relative[9] * J_TO_AU / GAS_CONSTANT / options.temperature)

            if options.spc is False:
                log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ") ", ))
                if options.QH and options.cosmo or options.solv:
                    log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} {:>13} '
                              '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)",
                                              'Solv-qh-G(T)'), thermodata=True)
                elif options.QH:
                    log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                              '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                              thermodata=True)
                elif options.cosmo or options.solv:
                    log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                              '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)",
                                              'Solv-qh-G(T)'), thermodata=True)
                else:
                    log.write('{:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                              '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                              thermodata=True)
            else:
                log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ") ", ))
                if options.QH and options.cosmo or options.QH and options.solv:
                    log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} {:>14} '
                              '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS", "T.qh-DS",
                                              "DG(T)_SPC", "qh-DG(T)_SPC", 'Solv-qh-G(T)_SPC'), thermodata=True)
                elif options.QH:
                    log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                              '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS", "T.qh-DS",
                                              "DG(T)_SPC", "qh-DG(T)_SPC"), thermodata=True)
                elif options.cosmo or options.solv:
                    log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                              '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS",
                                              "DG(T)_SPC", "qh-DG(T)_SPC", 'Solv-qh-G(T)_SPC'), thermodata=True)
                else:
                    log.write('{:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>14} '
                              '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS", "DG(T)_SPC",
                                              "qh-DG(T)_SPC"), thermodata=True)

            log.write("\n" + stars)

            for j, e_abs in enumerate(pes.e_abs[i]):
                if options.QH:
                    species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                               pes.qh_abs[i][j], options.temperature * pes.s_abs[i][j],
                               options.temperature * pes.qs_abs[i][j], pes.g_abs[i][j], pes.qhg_abs[i][j]]
                else:
                    species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                               options.temperature * pes.s_abs[i][j], options.temperature * pes.qs_abs[i][j],
                               pes.g_abs[i][j], pes.qhg_abs[i][j]]
                if options.cosmo or options.solv:
                    species.append(pes.solv_qhg_abs[i][j])
                relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                if pes.units == 'kJ/mol':
                    formatted_list = [J_TO_AU / 1000.0 * x for x in relative]
                else:
                    formatted_list = [KCAL_TO_AU * x for x in relative]  # Defaults to kcal/mol
                log.write("\no  ")
                if options.spc is False:
                    formatted_list = formatted_list[1:]
                    if options.QH and options.cosmo or options.solv:
                        if pes.dec == 1:
                            log.write('{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                      '{:13.1f} {:13.1f}'.format(pes.species[i][j], *formatted_list),
                                      thermodata=True)
                        if pes.dec == 2:
                            log.write('{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                      '{:13.2f} {:13.2f}'.format(pes.species[i][j], *formatted_list),
                                      thermodata=True)
                    elif options.QH or options.cosmo or options.solv:
                        if pes.dec == 1:
                            log.write('{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                      '{:13.1f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                        if pes.dec == 2:
                            log.write('{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                      '{:13.2f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                    else:
                        if pes.dec == 1:
                            log.write('{:<39} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                      '{:13.1f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                        if pes.dec == 2:
                            log.write('{:<39} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                      '{:13.2f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                else:
                    if options.QH and options.cosmo or options.QH and options.solv:
                        if pes.dec == 1:
                            log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} '
                                      '{:13.1f} {:13.1f} {:13.1f}'.format(pes.species[i][j], *formatted_list),
                                      thermodata=True)
                        if pes.dec == 2:
                            log.write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} '
                                      '{:13.2f} {:13.2f} {:13.2f}'.format(pes.species[i][j], *formatted_list),
                                      thermodata=True)
                    elif options.QH or options.cosmo or options.solv:
                        if pes.dec == 1:
                            log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} '
                                      '{:13.1f} {:13.1f}'.format(pes.species[i][j], *formatted_list),
                                      thermodata=True)
                        if pes.dec == 2:
                            log.write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} '
                                      '{:13.2f} {:13.2f}'.format(pes.species[i][j], *formatted_list),
                                      thermodata=True)
                    else:
                        if pes.dec == 1:
                            log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                      '{:13.1f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                        if pes.dec == 2:
                            log.write('{:<39} {:13.2f} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                      '{:13.2f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                if pes.boltz:
                    boltz = [math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / options.temperature) / e_sum,
                             math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / options.temperature) / h_sum,
                             math.exp(-relative[6] * J_TO_AU / GAS_CONSTANT / options.temperature) / g_sum,
                             math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / options.temperature) / qhg_sum]
                    selectivity = [boltz[x] * 100.0 for x in range(len(boltz))]
                    log.write("\n  " + '{:<39} {:13.2f}%{:24.2f}%{:35.2f}%{:13.2f}%'.format('', *selectivity))
                    sels.append(selectivity)
                formatted_list = [round(formatted_list[x], 6) for x in range(len(formatted_list))]
            if pes.boltz == 'ee' and len(sels) == 2:
                ee = [sels[0][x] - sels[1][x] for x in range(len(sels[0]))]
                if options.spc is False:
                    log.write("\n" + stars + "\n   " + '{:<39} {:13.1f}%{:24.1f}%{:35.1f}%{:13.1f}%'.format('ee (%)', *ee))
                else:
                    log.write("\n" + stars + "\n   " + '{:<39} {:27.1f} {:24.1f} {:35.1f} {:13.1f} '.format('ee (%)', *ee))
            log.write("\n" + stars + "\n")

def get_vib_scale_factor(level_of_theory, options, log):
    ''' Attempt to automatically obtain frequency scale factor
    Application of freq scale factors requires all outputs to be same level of theory'''

    if options.freq_scale_factor is not False:
        if 'ONIOM' not in level_of_theory[0]:
            log.write("\n\n   User-defined vibrational scale factor " + str(options.freq_scale_factor) + " for " +
                      level_of_theory[0] + " level of theory")
        else:
            log.write("\n\n   User-defined vibrational scale factor " + str(options.freq_scale_factor) +
                      " for QM region")

    else:
        # Look for vibrational scaling factor automatically
        if all_same(level_of_theory):
            level = level_of_theory[0].upper()

            for data in (scaling_data_dict, scaling_data_dict_mod):
                if level in data:

                    options.freq_scale_factor = data[level].zpe_fac
                    ref = scaling_refs[data[level].zpe_ref]
                    log.write("\n\no  Found vibrational scaling factor of {:.3f} for {} level of theory\n"
                              "   {}".format(options.freq_scale_factor, level_of_theory[0], ref))
                    break
        else:  # Print files and different levels of theory found
            files_l_o_t, levels_l_o_t, filtered_calcs_l_o_t = [], [], []
            for file in files:
                files_l_o_t.append(file)
            for i in l_o_t:
                levels_l_o_t.append(i)
            filtered_calcs_l_o_t.append(files_l_o_t)
            filtered_calcs_l_o_t.append(levels_l_o_t)
            print_check_fails(log, filtered_calcs_l_o_t[1], filtered_calcs_l_o_t[0], "levels of theory")

    # Exit program if molecular mechanics scaling factor is given and all files are not ONIOM calculations
    if options.mm_freq_scale_factor is not False:
        if all_same(l_o_t) and 'ONIOM' in l_o_t[0]:
            log.write("\n\no  User-defined vibrational scale factor " +
                      str(options.mm_freq_scale_factor) + " for MM region of " + l_o_t[0])
            log.write("\n   REF: {}".format(oniom_scale_ref))
        else:
            sys.exit("\n   Option --vmm is only for use in ONIOM calculation output files.\n   "
                     " help use option '-h'\n")

    if options.freq_scale_factor is False:
        options.freq_scale_factor = 1.0  # If no scaling factor is found use 1.0
        if all_same(level_of_theory):
            log.write("\n\n   Using vibrational scale factor {} for {} level of "
                      "theory".format(options.freq_scale_factor, level_of_theory[0]))
        else:
            log.write("\n\n   Using vibrational scale factor {}: differing levels of theory "
                      "detected.".format(options.freq_scale_factor))

    return options.freq_scale_factor, options.mm_freq_scale_factor

def main():
    files, bbe_vals = [], []

    # Get command line inputs. Use -h to list all possible arguments and default values
    parser = ArgumentParser()
    parser.add_argument("-q", dest="Q", action="store_true", default=False,
                        help="Quasi-harmonic entropy correction and enthalpy correction applied (default S=Grimme, "
                             "H=Head-Gordon)")
    parser.add_argument("--qs", dest="QS", default="grimme", type=str.lower, metavar="QS",
                        choices=('grimme', 'truhlar'),
                        help="Type of quasi-harmonic entropy correction (Grimme or Truhlar) (default Grimme)", )
    parser.add_argument("--qh", dest="QH", action="store_true", default=False,
                        help="Type of quasi-harmonic enthalpy correction (Head-Gordon)")
    parser.add_argument("-f", dest="freq_cutoff", default=100, type=float, metavar="FREQ_CUTOFF",
                        help="Cut-off frequency for both entropy and enthalpy (wavenumbers) (default = 100)", )
    parser.add_argument("--fs", dest="S_freq_cutoff", default=100.0, type=float, metavar="S_FREQ_CUTOFF",
                        help="Cut-off frequency for entropy (wavenumbers) (default = 100)")
    parser.add_argument("--fh", dest="H_freq_cutoff", default=100.0, type=float, metavar="H_FREQ_CUTOFF",
                        help="Cut-off frequency for enthalpy (wavenumbers) (default = 100)")
    parser.add_argument("-t", dest="temperature", default=298.15, type=float, metavar="TEMP",
                        help="Temperature (K) (default 298.15)")
    parser.add_argument("-c", dest="conc", default=False, type=float, metavar="CONC",
                        help="Concentration (mol/l) (default 1 atm)")
    parser.add_argument("--ti", dest="temperature_interval", default=False, metavar="TI",
                        help="Initial temp, final temp, step size (K)")
    parser.add_argument("-v", dest="freq_scale_factor", default=False, type=float, metavar="SCALE_FACTOR",
                        help="Frequency scaling factor. If not set, try to find a suitable value in database. "
                             "If not found, use 1.0")
    parser.add_argument("--vmm", dest="mm_freq_scale_factor", default=False, type=float, metavar="MM_SCALE_FACTOR",
                        help="Additional frequency scaling factor used in ONIOM calculations")
    parser.add_argument("--spc", dest="spc", type=str, default=False, metavar="SPC",
                        help="Indicates single point corrections (default False)")
    parser.add_argument("--solv", dest="solv", type=str, default=False, metavar="solv",
                        help="Indicates solvation correction (default False)")
    parser.add_argument("--boltz", dest="boltz", action="store_true", default=False,
                        help="Show Boltzmann factors")
    parser.add_argument("--cpu", dest="cputime", action="store_true", default=False,
                        help="Total CPU time")
    parser.add_argument("--d3", dest="D3", action="store_true", default=False,
                        help="Zero-damped DFTD3 correction will be computed")
    parser.add_argument("--d3bj", dest="D3BJ", action="store_true", default=False,
                        help="Becke-Johnson damped DFTD3 correction will be computed")
    parser.add_argument("--atm", dest="ATM", action="store_true", default=False,
                        help="Axilrod-Teller-Muto 3-body dispersion correction will be computed")
    parser.add_argument("--xyz", dest="xyz", action="store_true", default=False,
                        help="Write Cartesians to a .xyz file (default False)")
    parser.add_argument("--csv", dest="csv", action="store_true", default=False,
                        help="Write .csv output file format")
    parser.add_argument("--imag", dest="imag_freq", action="store_true", default=False,
                        help="Print imaginary frequencies (default False)")
    parser.add_argument("--invertifreq", dest="invert", nargs='?', const=True, default=False,
                        help="Make low lying imaginary frequencies positive (cutoff > -50.0 wavenumbers)")
    parser.add_argument("--freespace", dest="freespace", default="none", type=str, metavar="FREESPACE",
                        help="Solvent (H2O, toluene, DMF, AcOH, chloroform) (default none)")
    parser.add_argument("--dup", dest="duplicate", action="store_true", default=False,
                        help="Remove possible duplicates from thermochemical analysis")
    parser.add_argument("--cosmo", dest="cosmo", default=False, metavar="COSMO-RS",
                        help="Filename of a COSMO-RS .tab output file")
    parser.add_argument("--cosmo_int", dest="cosmo_int", default=False, metavar="COSMO-RS",
                        help="Filename of a COSMO-RS .tab output file along with a temperature range (K): "
                             "file.tab,'Initial_T, Final_T'")
    parser.add_argument("--output", dest="output", default="output", metavar="OUTPUT",
                        help="Change the default name of the output file to GoodVibes_\"output\".dat")
    parser.add_argument("--pes", dest="pes", default=False, metavar="PES",
                        help="Tabulate relative values")
    parser.add_argument("--nogconf", dest="gconf", action="store_false", default=True,
                        help="Calculate a free-energy correction related to multi-configurational space (default "
                             "calculate Gconf)")
    parser.add_argument("--ee", "--er", "--dr", dest="ee", default=False, type=str,
                        help="Tabulate selectivity values (excess, ratio) from a mixture, provide pattern for two "
                             "types such as *_R*,*_S*")
    parser.add_argument("--check", dest="check", action="store_true", default=False,
                        help="Checks if calculations were done with the same program, level of theory and solvent, "
                             "as well as detects potential duplicates")
    parser.add_argument("--media", dest="media", default=False, metavar="MEDIA",
                        help="Entropy correction for standard concentration of solvents")
    parser.add_argument("--custom_ext", type=str, default='',
                        help="List of additional file extensions to support, beyond .log or .out, use separated by "
                             "commas (ie, '.qfi, .gaussian'). It can also be specified with environment variable "
                             "GOODVIBES_CUSTOM_EXT")
    parser.add_argument("--graph", dest='graph', default=False, metavar="GRAPH",
                        help="Graph a reaction profile based on free energies calculated. ")
    parser.add_argument("--ssymm", dest='ssymm', action="store_true", default=False,
                        help="Turn on the symmetry correction.")
    parser.add_argument("--clustering", dest='clustering', action="store_true", default=False,
                        help="Turn on clustering.")

    # Parse Arguments
    (options, args) = parser.parse_known_args()
    options.command = '   Requested: '
    options.start = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())

    # If requested, turn on head-gordon enthalpy correction
    if options.Q: options.QH = True

    # If user has specified different file extensions
    if options.custom_ext or os.environ.get('GOODVIBES_CUSTOM_EXT', ''):
        custom_extensions = options.custom_ext.split(',') + os.environ.get('GOODVIBES_CUSTOM_EXT', '').split(',')
        for ext in custom_extensions:
            SUPPORTED_EXTENSIONS.add(ext.strip())

    # Default value for inverting imaginary frequencies
    if options.invert: options.invert == -50.0
    elif options.invert > 0: options.invert = -1 * options.invert

    # whether to group conformers
    clusters = []
    if options.clustering: options.command += '(clustering active)'

    # Start a log for the results
    log = Logger("Goodvibes", options.output, options.csv)

    # figure out whether clustering is required
    if len(args) > 1:
        for elem in args:
            if elem == 'clust:':
                options.clustering, options.boltz, nclust = True, True, -1

    # Get the filenames from the command line prompt
    args = sys.argv[1:]
    for elem in args:
        if options.clustering:
            if elem == 'clust:':
                clusters.append([])
                nclust += 0
        try:
            if os.path.splitext(elem)[1].lower() in SUPPORTED_EXTENSIONS:  # Look for file names
                for file in glob(elem):
                    if options.spc is False:
                        if file is not options.cosmo or options.solv:
                            files.append(file)
                        if options.clustering:
                            clusters[nclust].append(file)
                    else:
                        if file.find(options.spc + ".") == -1:
                            files.append(file)
                            if options.clustering:
                                clusters[nclust].append(file)

            elif elem != 'clust:':  # Look for requested options
                options.command += elem + ' '
        except IndexError:
            pass

    # Initial read of files - figure out what type they are
    file_data, kwargs = [], {}
    for i, file in enumerate(files):
        cc_data = ccread(file, **kwargs)

        if hasattr(cc_data, 'metadata'):
            if cc_data.metadata['package'] != 'Gaussian':
                log.write('x  ' + file + ' format ' + cc_data.metadata['package'] +' not yet supported!')
            elif cc_data.metadata['success'] != True:
                log.write('\nx  ' + file + ' did not terminate normally!')
            else:
                cc_data.name = os.path.splitext(file)[0]
                try: cc_data.metadata["functional"]#, cc_data.metadata['basis_set'])
                except KeyError: cc_data.metadata["functional"] = 'unknown'
                try: cc_data.metadata["basis_set"]#, cc_data.metadata['basis_set'])
                except KeyError: cc_data.metadata["basis_set"] = 'unknown'
                log.write('\no  ' + file + ': ' + cc_data.metadata['package'] +' '+cc_data.metadata['functional']+'/'+cc_data.metadata['basis_set']+' job terminated normally')
                file_data.append(cc_data)

    # Check if user has specified any files, if not quit now
    if len(files) == 0 or len(file_data) == 0:
        sys.exit("\nNo output files found!\n"
                 "For help with GoodVibes, use option '-h'\n")

    # Check the level of theory is consistent across all files
    try:
        level_of_theory = [file.metadata["functional"] + '/' + file.metadata['basis_set'] for file in file_data]
        implicit_solvation = [file.solvation for file in file_data]
        if all_same(level_of_theory):
            log.write('\no  All jobs performed at the ' + level_of_theory[-1] + ' level of theory')

        # Exit program if a comparison of Boltzmann factors is requested and level of theory is not uniform across all files
        if not all_same(level_of_theory) and (options.boltz is not False or options.ee is not False):
            sys.exit("\n\nERROR: When comparing files with Boltzmann factors (with bolts, ee, dr options), the level of "
             "theory used should be the same for all files.\n ")
    except ValueError: pass

    # Checks to see whether the available free space of a requested solvent is defined
    freespace = get_free_space(options.freespace)
    if freespace != 1000.0:
        log.write("\n   Specified solvent " + options.freespace + ": free volume " + str(
            "%.3f" % (freespace / 10.0)) + " (mol/l) corrects the translational entropy")

    # Check for implicit solvation
    printed_solv_warn = False
    try:
        for solv in implicit_solvation:
            if ('smd' in solv[0].lower() or 'pcm' in solv[0].lower()) and not printed_solv_warn:
                log.write("\n   Implicit solvation (SMD/CPCM) detected. Enthalpic and entropic terms are not separable "
                      "safely separated. Use them at your own risk!")
                printed_solv_warn = True
    except ValueError: pass

    # COSMO-RS temperature interval
    if options.cosmo_int:
        args = options.cosmo_int.split(',')
        cfile = args[0]
        cinterval = args[1:]
        log.write('\n\n   Reading COSMO-RS file: ' + cfile + ' over a T range of ' + cinterval[0] + '-' +
                  cinterval[1] + ' K.')

        t_interval, gsolv_dicts = cosmo_rs_out(cfile, files, interval=cinterval)
        options.temperature_interval = True

    elif options.cosmo is not False:  # Read from COSMO-RS output
        try:
            cosmo_solv = cosmo_rs_out(options.cosmo, files)
            log.write('\n\n   Reading COSMO-RS file: ' + options.cosmo)
        except ValueError:
            cosmo_solv = None
            log.write('\n\n   Warning! COSMO-RS file ' + options.cosmo + ' requested but not found')

    if options.freq_cutoff != 100.0:
        options.S_freq_cutoff = options.freq_cutoff
        options.H_freq_cutoff = options.freq_cutoff

    # Look up vibration scaling factor if not already supplied
    if all_same(level_of_theory):
        options.freq_scale_factor, options.mm_freq_scale_factor = get_vib_scale_factor(level_of_theory, options, log)
    else:
        options.freq_scale_factor = 1.0

    # If not at standard temp, need to correct the molarity of 1 atmosphere (assuming pressure is still 1 atm)
    if options.conc:
        options.gas_phase = False
    else:
        options.gas_phase = True
        options.conc = ATMOS / (GAS_CONSTANT * options.temperature)

    # Check for special options
    inverted_freqs, inverted_files = [], []
    if options.ssymm:
        ssymm_option = options.ssymm
    else:
        ssymm_option = False
    if options.mm_freq_scale_factor is not False:
        vmm_option = options.mm_freq_scale_factor
    else:
        vmm_option = False

    # Loop over all specified output files and grab COSMO data
    for n, file in enumerate(files):
        if options.cosmo:
            cosmo_option = cosmo_solv[file]
        else:
            cosmo_option = None

    # this is the actual thermochemistry calculation!
    bbe_vals = [calc_bbe(file, options, cosmo=cosmo_option, ssymm=ssymm_option, mm_freq_scale_factor=vmm_option) for file in file_data]
    species = [file.name for file in file_data]

    # Creates a new dictionary object thermo_data, which attaches the bbe data to each file-name
    thermo_data = dict(zip(species, bbe_vals))  # The collected thermochemical data for all files
    interval_bbe_data, interval_thermo_data = [], []

    inverted_freqs, inverted_files = [], []
    for file in species:
        if len(thermo_data[file].inverted_freqs) > 0:
            inverted_freqs.append(thermo_data[file].inverted_freqs)
            inverted_files.append(file)

    # Check if user has chosen to make any low lying imaginary frequencies positive
    if options.invert is not False:
        for i, file in enumerate(inverted_files):
            log.write("\n\n!  The following frequencies were made positive and used in calculations: " +
                          str(inverted_freqs[i]) + " from " + file)

    # Standard job (single temperature) requested
    intro = print_intro(options, log)

    if options.temperature_interval is False:
        # Look for duplicates or enantiomers if requested
        if options.duplicate: dup_list = check_dup(species, thermo_data, log)
        else: dup_list = []

        # Printing results
        summary = print_main(species, thermo_data, options, dup_list, clusters, log)

    # If necessary, create a file with Cartesians
    if options.xyz:
        xyz = xyz_out("Goodvibes_output.xyz", file_data)

    # Running a variable temperature analysis of the enthalpy, entropy and the free energy
    elif options.temperature_interval:
        stars = "   " + "*" * 128
        log.write("\n\n   Variable-Temperature analysis of the enthalpy, entropy and the entropy at a constant pressure between")
        if options.cosmo_int is False:
            temperature_interval = [float(temp) for temp in options.temperature_interval.split(',')]
            # If no temperature step was defined, divide the region into 10
            if len(temperature_interval) == 2:
                temperature_interval.append((temperature_interval[1] - temperature_interval[0]) / 10.0)
            interval = range(int(temperature_interval[0]), int(temperature_interval[1] + 1),
                             int(temperature_interval[2]))
            log.write("\n   T init:  %.1f,  T final:  %.1f,  T interval: %.1f" % (
                temperature_interval[0], temperature_interval[1], temperature_interval[2]))
        else:
            interval = t_interval
            log.write("\n   T init:  %.1f,   T final: %.1f" % (interval[0], interval[-1]))

        if options.QH:
            qh_print_format = "\n\n   {:<39} {:>13} {:>24} {:>13} {:>10} {:>10} {:>13} {:>13}"
            if options.spc and options.cosmo_int:
                log.write(qh_print_format.format("Structure", "Temp/K", "H_SPC", "qh-H_SPC", "T.S", "T.qh-S",
                                                 "G(T)_SPC", "Solv-qh-G(T)_SPC"), thermodata=True)
            elif options.cosmo_int:
                log.write(qh_print_format.format("Structure", "Temp/K", "H", "qh-H", "T.S", "T.qh-S", "G(T)",
                                                 "qh-G(T)", "Solv-qh-G(T)"), thermodata=True)
            elif options.spc:
                log.write(qh_print_format.format("Structure", "Temp/K", "H_SPC", "qh-H_SPC", "T.S", "T.qh-S",
                                                 "G(T)_SPC", "qh-G(T)_SPC"), thermodata=True)
            else:
                log.write(qh_print_format.format("Structure", "Temp/K", "H", "qh-H", "T.S", "T.qh-S", "G(T)",
                                                 "qh-G(T)"), thermodata=True)
        else:
            print_format_3 = '\n\n   {:<39} {:>13} {:>24} {:>10} {:>10} {:>13} {:>13}'
            if options.spc and options.cosmo_int:
                log.write(print_format_3.format("Structure", "Temp/K", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC",
                                                "Solv-qh-G(T)_SPC"), thermodata=True)
            elif options.cosmo_int:
                log.write(print_format_3.format("Structure", "Temp/K", "H", "T.S", "T.qh-S", "G(T)", "qh-G(T)",
                                                "Solv-qh-G(T)"), thermodata=True)
            elif options.spc:
                log.write(print_format_3.format("Structure", "Temp/K", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC",
                                                "qh-G(T)_SPC"), thermodata=True)
            else:
                log.write(print_format_3.format("Structure", "Temp/K", "H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),
                          thermodata=True)

        for h, file in enumerate(files):  # Temperature interval
            log.write("\n" + stars)
            interval_bbe_data.append([])
            for i in range(len(interval)):  # Iterate through the temperature range
                temp = interval[i]
                if options.gas_phase:
                    conc = ATMOS / GAS_CONSTANT / temp
                else:
                    conc = options.conc
                linear_warning = []
                if options.cosmo_int is False:
                    cosmo_option = False
                else:
                    cosmo_option = gsolv_dicts[i][file]
                if options.cosmo_int is False:
                    # haven't implemented D3 for this option
                    file_data = ccread(file, **kwargs)
                    options.temperature, options.conc = temp, conc
                    bbe = calc_bbe(file_data, options, cosmo=cosmo_option)
                interval_bbe_data[h].append(bbe)
                linear_warning.append(bbe.linear_warning)
                if linear_warning == [['Warning! Potential invalid calculation of linear molecule from Gaussian.']]:
                    log.write("\nx  ")
                    log.write('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]), thermodata=True)
                    log.write('             Warning! Potential invalid calculation of linear molecule from Gaussian ...')
                else:
                    # Gaussian spc files
                    if hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                        log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                    # ORCA spc files
                    elif not hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                        log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                    if not hasattr(bbe, "gibbs_free_energy"):
                        log.write("Warning! Couldn't find frequency information ...")
                    else:
                        log.write("\no  ")
                        log.write('{:<39} {:13.1f}'.format(os.path.splitext(os.path.basename(file))[0], temp),
                                  thermodata=True)
                        if not options.media:
                            if all(getattr(bbe, attrib) for attrib in
                                   ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                                if options.QH:
                                    if options.cosmo_int:
                                        log.write(' {:24.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(
                                            bbe.enthalpy, bbe.qh_enthalpy, (temp * bbe.entropy),
                                            (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.solv_qhg),
                                            thermodata=True)
                                    else:
                                        log.write(' {:24.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(
                                            bbe.enthalpy, bbe.qh_enthalpy, (temp * bbe.entropy),
                                            (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                            thermodata=True)
                                else:
                                    if options.cosmo_int:
                                        log.write(' {:24.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.enthalpy, (
                                                temp * bbe.entropy), (temp * bbe.qh_entropy), bbe.gibbs_free_energy,
                                                                                                         bbe.solv_qhg),
                                                  thermodata=True)
                                    else:
                                        log.write(' {:24.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.enthalpy, (
                                                temp * bbe.entropy), (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                                  thermodata=True)
                        else:
                            try:
                                from .media import solvents
                            except:
                                from media import solvents
                            if options.media.lower() in solvents and options.media.lower() == \
                                    os.path.splitext(os.path.basename(file))[0].lower():
                                mw_solvent = solvents[options.media.lower()][0]
                                density_solvent = solvents[options.media.lower()][1]
                                concentration_solvent = (density_solvent * 1000) / mw_solvent
                                media_correction = -(GAS_CONSTANT / J_TO_AU) * math.log(concentration_solvent)
                                if all(getattr(bbe, attrib) for attrib in
                                       ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy",
                                        "qh_gibbs_free_energy"]):
                                    if options.QH:
                                        log.write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                                  '{:13.6f}'.format(bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy,
                                                                    (temp * (bbe.entropy + media_correction)),
                                                                    (temp * (bbe.qh_entropy + media_correction)),
                                                                    bbe.gibbs_free_energy + (temp * (-media_correction)),
                                                                    bbe.qh_gibbs_free_energy + (temp * (-media_correction))))
                                        log.write("  Solvent")
                                else:
                                    log.write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                              '{:13.6f}'.format(bbe.zpe, bbe.enthalpy,
                                                                (temp * (bbe.entropy + media_correction)),
                                                                (temp * (bbe.qh_entropy + media_correction)),
                                                                bbe.gibbs_free_energy + (temp * (-media_correction)),
                                                                bbe.qh_gibbs_free_energy + (
                                                                            temp * (-media_correction))))
                                    log.write("  Solvent")
                            else:
                                if all(getattr(bbe, attrib) for attrib in
                                       ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy",
                                        "qh_gibbs_free_energy"]):
                                    if options.QH:
                                        log.write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                                  '{:13.6f}'.format(bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy,
                                                                    (temp * bbe.entropy), (temp * bbe.qh_entropy),
                                                                    bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy))
                                    else:
                                        log.write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                                  '{:13.6f}'.format(bbe.zpe, bbe.enthalpy, (temp * bbe.entropy),
                                                                    (temp * bbe.qh_entropy), bbe.gibbs_free_energy,
                                                                    bbe.qh_gibbs_free_energy))
            log.write("\n" + stars + "\n")

    # Perform checks for consistent options provided in calculation files (level of theory)
    if options.check:
        check_files(file_data, thermo_data, options, log)

    # Print CPU usage if requested
    if options.cputime:
        cpu = calc_cpu(species, thermo_data, options, log)

    # Tabulate relative values
    if options.pes:
        tabulate(thermo_data, options, log)

    # Compute enantiomeric excess
    if options.ee is not False:
        boltz_facs, weighted_free_energy, boltz_sum = get_boltz(species, thermo_data, options, clusters, dup_list)
        ee, er, ratio, dd_free_energy, failed, preference = get_selectivity(species, options, dup_list, boltz_facs, boltz_sum, log)

    # Graph reaction profiles
    if options.graph is not False:
        graph_data = get_pes(thermo_data, options, log)
        graph_reaction_profile(graph_data, options, log)

    # Close the log
    log.finalize()

if __name__ == "__main__":
    main()
