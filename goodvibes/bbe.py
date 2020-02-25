import os
import ctypes
import sys
import math
from . import parse
from . import energies
from . import entropies
from . import constants
from . import utils
from .get_out_data import getoutData


# The function to compute the "black box" entropy and enthalpy values
# along with all other thermochemical quantities
class calc_bbe:
    def __init__(
        self,
        file,
        QS,
        QH,
        s_freq_cutoff,
        H_FREQ_CUTOFF,
        temperature,
        conc,
        freq_scale_factor,
        solv,
        spc,
        invert,
        d3_term,
        ssymm=False,
        cosmo=None,
        mm_freq_scale_factor=False,
    ):
        # List of frequencies and default values
        (
            im_freq_cutoff,
            frequency_wn,
            im_frequency_wn,
            rotemp,
            roconst,
            linear_mol,
            link,
            freqloc,
            linkmax,
            symmno,
            self.cpu,
            inverted_freqs,
        ) = (
            0.0,
            [],
            [],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            0,
            0,
            0,
            0,
            1,
            [0, 0, 0, 0, 0],
            [],
        )
        linear_warning = False
        if mm_freq_scale_factor is False:
            fract_modelsys = False
        else:
            fract_modelsys = []
            freq_scale_factor = [freq_scale_factor, mm_freq_scale_factor]
        self.xyz = getoutData(file)
        self.job_type = parse.jobtype(file)
        # Parse some useful information from the file
        (
            self.sp_energy,
            self.program,
            self.version_program,
            self.solvation_model,
            self.file,
            self.charge,
            self.empirical_dispersion,
            self.multiplicity,
        ) = parse.parse_data(file)
        with open(file) as f:
            g_output = f.readlines()
        self.cosmo_qhg = 0.0
        # Read any single point energies if requested
        if spc != False and spc != "link":
            name, ext = os.path.splitext(file)
            try:
                (
                    self.sp_energy,
                    self.sp_program,
                    self.sp_version_program,
                    self.sp_solvation_model,
                    self.sp_file,
                    self.sp_charge,
                    self.sp_empirical_dispersion,
                    self.sp_multiplicity,
                ) = parse.parse_data(name + "_" + spc + ext)
                self.cpu = parse.sp_cpu(name + "_" + spc + ext)
            except ValueError:
                self.sp_energy = "!"
                pass
        elif spc == "link":
            (
                self.sp_energy,
                self.sp_program,
                self.sp_version_program,
                self.sp_solvation_model,
                self.sp_file,
                self.sp_charge,
                self.sp_empirical_dispersion,
                self.sp_multiplicity,
            ) = parse.parse_data(file)
        # Count number of links
        for line in g_output:
            # Only read first link + freq not other link jobs
            if "Normal termination" in line:
                linkmax += 1
            else:
                frequency_wn = []
            if "Frequencies --" in line:
                freqloc = linkmax

        # Iterate over output
        if freqloc == 0:
            freqloc = len(g_output)
        for i, line in enumerate(g_output):
            # Link counter
            if "Normal termination" in line:
                link += 1
                # Reset frequencies if in final freq link
                if link == freqloc:
                    frequency_wn = []
                    im_frequency_wn = []
                    if mm_freq_scale_factor is not False:
                        fract_modelsys = []
            # If spc specified will take last Energy from file, otherwise will break after freq calc
            if link > freqloc:
                break
            # Iterate over output: look out for low frequencies
            if line.strip().startswith("Frequencies -- "):
                if mm_freq_scale_factor is not False:
                    newline = g_output[i + 3]
                for j in range(2, 5):
                    try:
                        x = float(line.strip().split()[j])
                        # If given MM freq scale factor fill the fract_modelsys array:
                        if mm_freq_scale_factor is not False:
                            y = float(newline.strip().split()[j]) / 100.0
                            y = float(f"{y:.6f}")
                        else:
                            y = 1.0
                        # Only deal with real frequencies
                        if x > 0.00:
                            frequency_wn.append(x)
                            if mm_freq_scale_factor is not False:
                                fract_modelsys.append(y)
                        # Check if we want to make any low lying imaginary frequencies positive
                        elif x < -1 * im_freq_cutoff:
                            if invert is not False:
                                if x > float(invert):
                                    frequency_wn.append(x * -1.0)
                                    inverted_freqs.append(x)
                                else:
                                    im_frequency_wn.append(x)
                            else:
                                im_frequency_wn.append(x)
                    except IndexError:
                        pass
            # For QM calculations look for SCF energies, last one will be the optimized energy
            elif line.strip().startswith("SCF Done:"):
                self.scf_energy = float(line.strip().split()[4])
            # For Counterpoise calculations the corrected energy value will be taken
            elif line.strip().startswith("Counterpoise corrected energy"):
                self.scf_energy = float(line.strip().split()[4])
            # For MP2 calculations replace with EUMP2
            elif "EUMP2 =" in line.strip():
                self.scf_energy = float((line.strip().split()[5]).replace("D", "E"))
            # For ONIOM calculations use the extrapolated value rather than SCF value
            elif "ONIOM: extrapolated energy" in line.strip():
                self.scf_energy = float(line.strip().split()[4])
            # For Semi-empirical or Molecular Mechanics calculations
            elif (
                "Energy= " in line.strip()
                and "Predicted" not in line.strip()
                and "Thermal" not in line.strip()
            ):
                self.scf_energy = float(line.strip().split()[1])
            # Look for thermal corrections, paying attention to point group symmetry
            elif line.strip().startswith("Zero-point correction="):
                self.zero_point_corr = float(line.strip().split()[2])
            # Grab Multiplicity
            elif "Multiplicity" in line.strip():
                try:
                    self.mult = int(line.split("=")[-1].strip().split()[0])
                except:
                    self.mult = int(line.split()[-1])
            # Grab molecular mass
            elif line.strip().startswith("Molecular mass:"):
                molecular_mass = float(line.strip().split()[2])
            # Grab rational symmetry number
            elif line.strip().startswith("Rotational symmetry number"):
                symmno = int((line.strip().split()[3]).split(".")[0])
            # Grab point group
            elif line.strip().startswith("Full point group"):
                if line.strip().split()[3] == "D*H" or line.strip().split()[3] == "C*V":
                    linear_mol = 1
            # Grab rotational constants
            elif line.strip().startswith("Rotational constants (GHZ):"):
                try:
                    self.roconst = [
                        float(line.strip().replace(":", " ").split()[3]),
                        float(line.strip().replace(":", " ").split()[4]),
                        float(line.strip().replace(":", " ").split()[5]),
                    ]
                except ValueError:
                    if line.strip().find("********"):
                        linear_warning = True
                        self.roconst = [
                            float(line.strip().replace(":", " ").split()[4]),
                            float(line.strip().replace(":", " ").split()[5]),
                        ]
            # Grab rotational temperatures
            elif line.strip().startswith("Rotational temperature "):
                rotemp = [float(line.strip().split()[3])]
            elif line.strip().startswith("Rotational temperatures"):
                try:
                    rotemp = [
                        float(line.strip().split()[3]),
                        float(line.strip().split()[4]),
                        float(line.strip().split()[5]),
                    ]
                except ValueError:
                    rotemp = None
                    if line.strip().find("********"):
                        linear_warning = True
                        rotemp = [
                            float(line.strip().split()[4]),
                            float(line.strip().split()[5]),
                        ]
            if "Job cpu time" in line.strip():
                days = int(line.split()[3]) + self.cpu[0]
                hours = int(line.split()[5]) + self.cpu[1]
                mins = int(line.split()[7]) + self.cpu[2]
                secs = 0 + self.cpu[3]
                msecs = int(float(line.split()[9]) * 1000.0) + self.cpu[4]
                self.cpu = [days, hours, mins, secs, msecs]
        self.inverted_freqs = inverted_freqs
        # Skip the calculation if unable to parse the frequencies or zpe from the output file
        if hasattr(self, "zero_point_corr") and rotemp:
            cutoffs = [s_freq_cutoff for freq in frequency_wn]

            # Translational and electronic contributions to the energy and entropy do not depend on frequencies
            u_trans = energies.calc_translational_energy(temperature)
            s_trans = entropies.calc_translational_entropy(
                molecular_mass, conc, temperature, solv
            )
            s_elec = entropies.calc_electronic_entropy(self.mult)

            # Rotational and Vibrational contributions to the energy entropy
            if len(frequency_wn) > 0:
                zpe = energies.calc_zeropoint_energy(
                    frequency_wn, freq_scale_factor, fract_modelsys
                )
                u_rot = energies.calc_rotational_energy(
                    self.zero_point_corr, symmno, temperature, linear_mol
                )
                u_vib = energies.calc_vibrational_energy(
                    frequency_wn, temperature, freq_scale_factor, fract_modelsys
                )
                s_rot = entropies.calc_rotational_entropy(
                    self.zero_point_corr, linear_mol, symmno, rotemp, temperature
                )

                # Calculate harmonic entropy, free-rotor entropy and damping function for each frequency
                Svib_rrho = entropies.calc_rrho_entropy(
                    frequency_wn, temperature, freq_scale_factor, fract_modelsys
                )

                if s_freq_cutoff > 0.0:
                    Svib_rrqho = entropies.calc_rrho_entropy(
                        cutoffs, temperature, freq_scale_factor, fract_modelsys
                    )
                Svib_free_rot = entropies.calc_freerot_entropy(
                    frequency_wn, temperature, freq_scale_factor, fract_modelsys
                )
                S_damp = entropies.calc_damp(frequency_wn, s_freq_cutoff)

                # check for qh
                if QH:
                    Uvib_qrrho = energies.calc_qRRHO_energy(
                        frequency_wn, temperature, freq_scale_factor
                    )
                    H_damp = entropies.calc_damp(frequency_wn, H_FREQ_CUTOFF)

                # Compute entropy (cal/mol/K) using the two values and damping function
                vib_entropy = []
                vib_energy = []
                for j in range(0, len(frequency_wn)):
                    # Entropy correction
                    if QS == "grimme":
                        vib_entropy.append(
                            Svib_rrho[j] * S_damp[j]
                            + (1 - S_damp[j]) * Svib_free_rot[j]
                        )
                    elif QS == "truhlar":
                        if s_freq_cutoff > 0.0:
                            if frequency_wn[j] > s_freq_cutoff:
                                vib_entropy.append(Svib_rrho[j])
                            else:
                                vib_entropy.append(Svib_rrqho[j])
                        else:
                            vib_entropy.append(Svib_rrho[j])
                    # Enthalpy correction
                    if QH:
                        vib_energy.append(
                            H_damp[j] * Uvib_qrrho[j]
                            + (1 - H_damp[j])
                            * 0.5
                            * constants.GAS_CONSTANT
                            * temperature
                        )

                qh_s_vib, h_s_vib = sum(vib_entropy), sum(Svib_rrho)
                if QH:
                    qh_u_vib = sum(vib_energy)
            else:
                zpe, u_rot, u_vib, qh_u_vib, s_rot, h_s_vib, qh_s_vib = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )

            # The D3 term is added to the energy term here. If not requested then this term is zero
            # It is added to the SPC energy if defined (instead of the SCF energy)
            if spc is False:
                self.scf_energy += d3_term
            else:
                self.sp_energy += d3_term

            # Add terms (converted to au) to get Free energy - perform separately
            # for harmonic and quasi-harmonic values out of interest
            self.enthalpy = (
                self.scf_energy
                + (u_trans + u_rot + u_vib + constants.GAS_CONSTANT * temperature)
                / constants.J_TO_AU
            )
            self.qh_enthalpy = 0.0
            if QH:
                self.qh_enthalpy = (
                    self.scf_energy
                    + (
                        u_trans
                        + u_rot
                        + qh_u_vib
                        + constants.GAS_CONSTANT * temperature
                    )
                    / constants.J_TO_AU
                )
            # Single point correction replaces energy from optimization with single point value
            if spc is not False:
                try:
                    self.enthalpy = self.enthalpy - self.scf_energy + self.sp_energy
                except TypeError:
                    pass
                if QH:
                    try:
                        self.qh_enthalpy = (
                            self.qh_enthalpy - self.scf_energy + self.sp_energy
                        )
                    except TypeError:
                        pass

            self.zpe = zpe / constants.J_TO_AU
            self.entropy = (s_trans + s_rot + h_s_vib + s_elec) / constants.J_TO_AU
            self.qh_entropy = (s_trans + s_rot + qh_s_vib + s_elec) / constants.J_TO_AU

            # Symmetry - entropy correction for molecular symmetry
            if ssymm:
                sym_entropy_correction, pgroup = self.sym_correction(
                    file.split(".")[0].replace("/", "_")
                )
                self.point_group = pgroup
                self.entropy += sym_entropy_correction
                self.qh_entropy += sym_entropy_correction

            # Calculate Free Energy
            if QH:
                self.gibbs_free_energy = self.enthalpy - temperature * self.entropy
                self.qh_gibbs_free_energy = (
                    self.qh_enthalpy - temperature * self.qh_entropy
                )
            else:
                self.gibbs_free_energy = self.enthalpy - temperature * self.entropy
                self.qh_gibbs_free_energy = (
                    self.enthalpy - temperature * self.qh_entropy
                )

            if cosmo:
                self.cosmo_qhg = self.qh_gibbs_free_energy + cosmo
            self.im_freq = []
            for freq in im_frequency_wn:
                if freq < -1 * im_freq_cutoff:
                    self.im_freq.append(freq)
        self.frequency_wn = frequency_wn
        self.im_frequency_wn = im_frequency_wn
        self.linear_warning = linear_warning

    # Get external symmetry number
    def ex_sym(self, file):
        coords_string = self.xyz.coords_string()
        coords = coords_string.encode("utf-8")
        c_coords = ctypes.c_char_p(coords)

        # Determine OS with sys.platform to see what compiled symmetry file to use
        platform = sys.platform
        if platform.startswith("linux"):  # linux - .so file
            path1 = utils.sharepath("symmetry_linux.so")
            newlib = "lib_" + file + ".so"
            path2 = utils.sharepath(newlib)
            copy = "cp " + path1 + " " + path2
            os.popen(copy).close()
            symmetry = ctypes.CDLL(path2)
        elif platform.startswith("darwin"):  # macOS - .dylib file
            path1 = utils.sharepath("symmetry_mac.dylib")
            newlib = "lib_" + file + ".dylib"
            path2 = utils.sharepath(newlib)
            copy = "cp " + path1 + " " + path2
            os.popen(copy).close()
            symmetry = ctypes.CDLL(path2)
        elif platform.startswith("win"):  # windows - .dll file
            path1 = utils.sharepath("symmetry_windows.dll")
            newlib = "lib_" + file + ".dll"
            path2 = utils.sharepath(newlib)
            copy = "copy " + path1 + " " + path2
            os.popen(copy).close()
            symmetry = ctypes.cdll.LoadLibrary(path2)

        symmetry.symmetry.restype = ctypes.c_char_p
        pgroup = symmetry.symmetry(c_coords).decode("utf-8")
        ex_sym = constants.pg_sm.get(pgroup)

        # Remove file
        if platform.startswith("linux"):  # linux - .so file
            remove = "rm " + path2
            os.popen(remove).close()
        elif platform.startswith("darwin"):  # macOS - .dylib file
            remove = "rm " + path2
            os.popen(remove).close()
        elif platform.startswith("win"):  # windows - .dll file
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
            if self.xyz.atom_nums[i] != 6:
                continue
            As = np.array(self.xyz.atom_nums)[row]
            if len(As == 4):
                neighbors = [x for x in As if x in neighbor]
                caps = [x for x in As if x in cap]
                if (len(neighbors) == 1) and (len(set(caps)) == 1):
                    int_sym *= 3
        return int_sym

    def sym_correction(self, file):
        ex_sym, pgroup = self.ex_sym(file)
        int_sym = self.int_sym()
        # override int_sym
        int_sym = 1
        sym_num = ex_sym * int_sym
        sym_correction = (
            -constants.GAS_CONSTANT * math.log(sym_num)
        ) / constants.J_TO_AU
        return sym_correction, pgroup
