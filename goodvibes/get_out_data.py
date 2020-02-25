from . import constants
from . import utils
import numpy as np


# Read molecule data from a compchem output file
# Currently supports Gaussian and ORCA output types
class getoutData:
    def __init__(self, file):
        with open(file) as f:
            data = f.readlines()
        program = "none"

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
                        for j in range(3, nfreqs + 1):
                            self.REDMASS.append(float(outlines[i + 1].split()[j]))
                        for j in range(3, nfreqs + 1):
                            self.FORCECONST.append(float(outlines[i + 2].split()[j]))

                        for j in range(0, natoms):
                            for k in range(0, nfreqs - 2):
                                self.NORMALMODE[(freqs_so_far + k)].append(
                                    [
                                        float(outlines[i + 5 + j].split()[3 * k + 2]),
                                        float(outlines[i + 5 + j].split()[3 * k + 3]),
                                        float(outlines[i + 5 + j].split()[3 * k + 4]),
                                    ]
                                )
                        freqs_so_far = freqs_so_far + nfreqs - 2

        def getatom_types(self, outlines, program):
            if program == "Gaussian":
                for i, oline in enumerate(outlines):
                    if "Input orientation" in oline or "Standard orientation" in oline:
                        (
                            self.atom_nums,
                            self.atom_types,
                            self.cartesians,
                            self.atomictypes,
                            carts,
                        ) = ([], [], [], [], outlines[i + 5 :])
                        for j, line in enumerate(carts):
                            if "-------" in line:
                                break
                            self.atom_nums.append(int(line.split()[1]))
                            self.atom_types.append(
                                utils.element_id(int(line.split()[1]))
                            )
                            self.atomictypes.append(int(line.split()[2]))
                            if len(line.split()) > 5:
                                self.cartesians.append(
                                    [
                                        float(line.split()[3]),
                                        float(line.split()[4]),
                                        float(line.split()[5]),
                                    ]
                                )
                            else:
                                self.cartesians.append(
                                    [
                                        float(line.split()[2]),
                                        float(line.split()[3]),
                                        float(line.split()[4]),
                                    ]
                                )
            if program == "Orca":
                for i, oline in enumerate(outlines):
                    if "*" in oline and ">" in oline and "xyz" in oline:
                        self.atom_nums, self.atom_types, self.cartesians, carts = (
                            [],
                            [],
                            [],
                            outlines[i + 1 :],
                        )
                        for j, line in enumerate(carts):
                            if ">" in line and "*" in line:
                                break
                            if len(line.split()) > 5:
                                self.cartesians.append(
                                    [
                                        float(line.split()[3]),
                                        float(line.split()[4]),
                                        float(line.split()[5]),
                                    ]
                                )
                                self.atom_types.append(line.split()[2])
                                self.atom_nums.append(
                                    utils.element_id(line.split()[2], num=True)
                                )
                            else:
                                self.cartesians.append(
                                    [
                                        float(line.split()[2]),
                                        float(line.split()[3]),
                                        float(line.split()[4]),
                                    ]
                                )
                                self.atom_types.append(line.split()[1])
                                self.atom_nums.append(
                                    utils.element_id(line.split()[1], num=True)
                                )

        getatom_types(self, data, program)
        natoms = len(self.atom_types)
        try:
            get_freqs(self, data, natoms, program)
        except:
            pass

    # Convert coordinates to string that can be used by the symmetry.c program
    def coords_string(self):
        xyzstring = str(len(self.atom_nums)) + "\n"
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
                cutoff = constants.RADII[ai] + constants.RADII[aj] + tolerance
                distance = np.linalg.norm(
                    np.array(self.cartesians[i]) - np.array(self.cartesians[j])
                )
                if distance < cutoff:
                    row.append(j)
            connectivity.append(row)
            self.connectivity = connectivity
