from datetime import datetime, timedelta
import sys
import os
from . import constants
from .get_out_data import getoutData
from . import parse


# Enables output to terminal and to text file
class Logger:
    def __init__(self, filein, append, csv):
        self.csv = csv
        if not self.csv:
            suffix = "dat"
        else:
            suffix = "csv"
        self.log = open("{0}_{1}.{2}".format(filein, append, suffix), "w")

    def write(self, message, thermodata=False):
        self.thermodata = thermodata
        print(message, end="")
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


# Enables output of optimized coordinates to a single xyz-formatted file
class xyz_out:
    def __init__(self, filein, suffix, append):
        self.xyz = open("{}_{}.{}".format(filein, append, suffix), "w")

    def write_text(self, message):
        self.xyz.write(message + "\n")

    def write_coords(self, atoms, coords):
        for n, carts in enumerate(coords):
            self.xyz.write("{:>1}".format(atoms[n]))
            for cart in carts:
                self.xyz.write("{:13.6f}".format(cart))
            self.xyz.write("\n")

    def finalize(self):
        self.xyz.close()


# Calculate elapsed time
def add_time(tm, cpu):
    [days, hrs, mins, secs, msecs] = cpu
    fulldate = datetime(100, 1, tm.day, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(
        days=days, hours=hrs, minutes=mins, seconds=secs, microseconds=msecs * 1000
    )
    return fulldate


def sharepath(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "share", filename)


def element_id(massno, num=False):
    try:
        if num:
            return constants.periodictable.index(massno)
        return constants.periodictable[massno]
    except IndexError:
        return "XX"


def all_same(items):
    return all(x == items[0] for x in items)


# Function for printing unique checks
def print_check_fails(log, check_attribute, file, attribute, option2=False):
    unique_attr = {}
    for i, attr in enumerate(check_attribute):
        if option2 is not False:
            attr = (attr, option2[i])
        if attr not in unique_attr:
            unique_attr[attr] = [file[i]]
        else:
            unique_attr[attr].append(file[i])
    log.write("\nx  Caution! Different {} found: ".format(attribute))
    for attr in unique_attr:
        if option2 is not False:
            if float(attr[0]) < 0:
                log.write("\n       {} {}: ".format(attr[0], attr[1]))
            else:
                log.write("\n        {} {}: ".format(attr[0], attr[1]))
        else:
            log.write("\n        -{}: ".format(attr))
        for filename in unique_attr[attr]:
            if filename is unique_attr[attr][-1]:
                log.write("{}".format(filename))
            else:
                log.write("{}, ".format(filename))


# Perform careful checks on calculation output files
# Check for Gaussian version, solvation state/gas phase consistency, level of theory/basis set consistency,
# charge and multiplicity consistency, standard concentration used, potential linear molecule error,
# transition state verification, empirical dispersion models.
def check_files(log, files, thermo_data, options, STARS, l_o_t, orientation, grid):
    log.write("\n   Checks for thermochemistry calculations (frequency calculations):")
    log.write("\n" + STARS)
    # Check program used and version
    version_check = [thermo_data[key].version_program for key in thermo_data]
    file_check = [thermo_data[key].file for key in thermo_data]
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
        if solvent_check[0] == "gas phase" and str(round(options.conc, 4)) == str(
            round(0.0408740470708, 4)
        ):
            log.write("\no  Using a standard concentration of 1 atm for gas phase.")
        elif solvent_check[0] == "gas phase" and str(round(options.conc, 4)) != str(
            round(0.0408740470708, 4)
        ):
            log.write(
                "\nx  Caution! Standard concentration is not 1 atm for gas phase (using {} M).".format(
                    options.conc
                )
            )
        elif solvent_check[0] != "gas phase" and str(round(options.conc, 4)) == str(
            round(0.0408740470708, 4)
        ):
            log.write(
                "\nx  Using a standard concentration of 1 atm for solvent phase (option -c 1 should be included for 1 M)."
            )
        elif solvent_check[0] != "gas phase" and str(options.conc) == str(1.0):
            log.write("\no  Using a standard concentration of 1 M for solvent phase.")
        elif (
            solvent_check[0] != "gas phase"
            and str(round(options.conc, 4)) != str(round(0.0408740470708, 4))
            and str(options.conc) != str(1.0)
        ):
            log.write(
                "\nx  Caution! Standard concentration is not 1 M for solvent phase (using {} M).".format(
                    options.conc
                )
            )
    if all_same(solvent_check) == False and "gas phase" in solvent_check:
        log.write(
            "\nx  Caution! The right standard concentration cannot be determined because the calculations use a combination of gas and solvent phases."
        )
    if all_same(solvent_check) == False and "gas phase" not in solvent_check:
        log.write(
            "\nx  Caution! Different solvents used, fix this issue and use option -c 1 for a standard concentration of 1 M."
        )

    # Check charge and multiplicity
    charge_check = [thermo_data[key].charge for key in thermo_data]
    multiplicity_check = [thermo_data[key].multiplicity for key in thermo_data]
    if all_same(charge_check) != False and all_same(multiplicity_check) != False:
        log.write(
            "\no  Using charge {} and multiplicity {} in all calculations.".format(
                charge_check[0], multiplicity_check[0]
            )
        )
    else:
        print_check_fails(
            log, charge_check, file_check, "charge and multiplicity", multiplicity_check
        )

    # Check for duplicate structures
    dup_list = parse.check_dup(files, thermo_data)
    if len(dup_list) == 0:
        log.write("\no  No duplicates or enantiomers found")
    else:
        log.write("\nx  Caution! Possible duplicates or enantiomers found:")
        for dup in dup_list:
            log.write("\n        {} and {}".format(dup[0], dup[1]))

    # Check for linear molecules with incorrect number of vibrational modes
    (
        linear_fails,
        linear_fails_atom,
        linear_fails_cart,
        linear_fails_files,
        linear_fails_list,
    ) = ([], [], [], [], [])
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
            if (
                linear_fails_list[0][i] == ["I", "I", "I"]
                or linear_fails_list[0][i] == ["O", "O", "O"]
                or linear_fails_list[0][i] == ["N", "N", "N"]
                or linear_fails_list[0][i] == ["H", "C", "N"]
                or linear_fails_list[0][i] == ["H", "N", "C"]
                or linear_fails_list[0][i] == ["C", "H", "N"]
                or linear_fails_list[0][i] == ["C", "N", "H"]
                or linear_fails_list[0][i] == ["N", "H", "C"]
                or linear_fails_list[0][i] == ["N", "C", "H"]
            ):
                if len(linear_fails_list[2][i]) == 4:
                    linear_mol_correct.append(linear_fails_list[3][i])
                else:
                    linear_mol_wrong.append(linear_fails_list[3][i])
            else:
                for j in range(len(linear_fails_list[0][i])):
                    for k in range(len(linear_fails_list[0][i])):
                        if k > j:
                            for l in range(len(linear_fails_list[1][i][j])):
                                if (
                                    linear_fails_list[0][i][j]
                                    == linear_fails_list[0][i][k]
                                ):
                                    if linear_fails_list[1][i][j][l] > (
                                        -linear_fails_list[1][i][k][l] - 0.1
                                    ) and linear_fails_list[1][i][j][l] < (
                                        -linear_fails_list[1][i][k][l] + 0.1
                                    ):
                                        count_linear = count_linear + 1
                                        if count_linear == 3:
                                            if len(linear_fails_list[2][i]) == 4:
                                                linear_mol_correct.append(
                                                    linear_fails_list[3][i]
                                                )
                                            else:
                                                linear_mol_wrong.append(
                                                    linear_fails_list[3][i]
                                                )
        if len(linear_fails_list[0][i]) == 4:
            if (
                linear_fails_list[0][i] == ["C", "C", "H", "H"]
                or linear_fails_list[0][i] == ["C", "H", "C", "H"]
                or linear_fails_list[0][i] == ["C", "H", "H", "C"]
                or linear_fails_list[0][i] == ["H", "C", "C", "H"]
                or linear_fails_list[0][i] == ["H", "C", "H", "C"]
                or linear_fails_list[0][i] == ["H", "H", "C", "C"]
            ):
                if len(linear_fails_list[2][i]) == 7:
                    linear_mol_correct.append(linear_fails_list[3][i])
                else:
                    linear_mol_wrong.append(linear_fails_list[3][i])
    linear_correct_print, linear_wrong_print = "", ""
    for i in range(len(linear_mol_correct)):
        linear_correct_print += ", " + linear_mol_correct[i]
    for i in range(len(linear_mol_wrong)):
        linear_wrong_print += ", " + linear_mol_wrong[i]
    linear_correct_print = linear_correct_print[1:]
    linear_wrong_print = linear_wrong_print[1:]
    if len(linear_mol_correct) == 0:
        if len(linear_mol_wrong) == 0:
            log.write("\n-  No linear molecules found.")
        if len(linear_mol_wrong) >= 1:
            log.write(
                "\nx  Caution! Potential linear molecules with wrong number of frequencies found "
                "(correct number = 3N-5) -{}.".format(linear_wrong_print)
            )
    elif len(linear_mol_correct) >= 1:
        if len(linear_mol_wrong) == 0:
            log.write(
                "\no  All the linear molecules have the correct number of frequencies -{}.".format(
                    linear_correct_print
                )
            )
        if len(linear_mol_wrong) >= 1:
            log.write(
                "\nx  Caution! Potential linear molecules with wrong number of frequencies found -{}. Correct "
                "number of frequencies (3N-5) found in other calculations -{}.".format(
                    linear_wrong_print, linear_correct_print
                )
            )

    # Checks whether any TS have > 1 imaginary frequency and any GS have any imaginary frequencies
    for file in files:
        bbe = thermo_data[file]
        if bbe.job_type.find("TS") > -1 and len(bbe.im_frequency_wn) != 1:
            log.write(
                "\nx  Caution! TS {} does not have 1 imaginary frequency greater than -50 wavenumbers.".format(
                    file
                )
            )
        if (
            bbe.job_type.find("GS") > -1
            and bbe.job_type.find("TS") == -1
            and len(bbe.im_frequency_wn) != 0
        ):
            log.write(
                "\nx  Caution: GS {} has 1 or more imaginary frequencies greater than -50 wavenumbers.".format(
                    file
                )
            )

    # Check for empirical dispersion
    dispersion_check = [thermo_data[key].empirical_dispersion for key in thermo_data]
    if all_same(dispersion_check):
        if dispersion_check[0] == "No empirical dispersion detected":
            log.write(
                "\n-  No empirical dispersion detected in any of the calculations."
            )
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
            if os.path.exists(name + "_" + options.spc + ".log"):
                names_spc.append(name + "_" + options.spc + ".log")
            elif os.path.exists(name + "_" + options.spc + ".out"):
                names_spc.append(name + "_" + options.spc + ".out")

        # Check SPC program versions
        version_check_spc = [thermo_data[key].sp_version_program for key in thermo_data]
        if all_same(version_check_spc):
            log.write(
                "\no  Using {} in all the single-point corrections.".format(
                    version_check_spc[0]
                )
            )
        else:
            print_check_fails(
                log, version_check_spc, file_check, "programs or versions"
            )

        # Check SPC solvation
        solvent_check_spc = [thermo_data[key].sp_solvation_model for key in thermo_data]
        if all_same(solvent_check_spc):
            if isinstance(solvent_check_spc[0], list):
                log.write(
                    "\no  Using "
                    + solvent_check_spc[0][0]
                    + " in all single-point corrections."
                )
            else:
                log.write(
                    "\no  Using "
                    + solvent_check_spc[0]
                    + " in all single-point corrections."
                )
        else:
            print_check_fails(log, solvent_check_spc, file_check, "solvation models")

        # Check SPC level of theory
        l_o_t_spc = [parse.level_of_theory(name) for name in names_spc]
        if all_same(l_o_t_spc):
            log.write(
                "\no  Using {} in all the single-point corrections.".format(
                    l_o_t_spc[0]
                )
            )
        else:
            print_check_fails(log, l_o_t_spc, file_check, "levels of theory")

        # Check SPC charge and multiplicity
        charge_spc_check = [thermo_data[key].sp_charge for key in thermo_data]
        multiplicity_spc_check = [
            thermo_data[key].sp_multiplicity for key in thermo_data
        ]
        if (
            all_same(charge_spc_check) != False
            and all_same(multiplicity_spc_check) != False
        ):
            log.write(
                "\no  Using charge and multiplicity {} {} in all the single-point corrections.".format(
                    charge_spc_check[0], multiplicity_spc_check[0]
                )
            )
        else:
            print_check_fails(
                log,
                charge_spc_check,
                file_check,
                "charge and multiplicity",
                multiplicity_spc_check,
            )

        # Check if the geometries of freq calculations match their corresponding structures in single-point calculations
        (
            geom_duplic_list,
            geom_duplic_list_spc,
            geom_duplic_cart,
            geom_duplic_files,
            geom_duplic_cart_spc,
            geom_duplic_files_spc,
        ) = ([], [], [], [], [], [])
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
                        elif "{0:.3f}".format(
                            geom_duplic_list[0][i][j][0]
                        ) == "{0:.3f}".format(
                            geom_duplic_list_spc[0][i][j][0] * (-1)
                        ) or "{0:.3f}".format(
                            geom_duplic_list[0][i][j][0]
                        ) == "{0:.3f}".format(
                            geom_duplic_list_spc[0][i][j][0]
                        ):
                            if "{0:.3f}".format(
                                geom_duplic_list[0][i][j][1]
                            ) == "{0:.3f}".format(
                                geom_duplic_list_spc[0][i][j][1] * (-1)
                            ) or "{0:.3f}".format(
                                geom_duplic_list[0][i][j][1]
                            ) == "{0:.3f}".format(
                                geom_duplic_list_spc[0][i][j][1] * (-1)
                            ):
                                count = count
                            if "{0:.3f}".format(
                                geom_duplic_list[0][i][j][2]
                            ) == "{0:.3f}".format(
                                geom_duplic_list_spc[0][i][j][2] * (-1)
                            ) or "{0:.3f}".format(
                                geom_duplic_list[0][i][j][2]
                            ) == "{0:.3f}".format(
                                geom_duplic_list_spc[0][i][j][2] * (-1)
                            ):
                                count = count
                        else:
                            spc_mismatching += ", " + geom_duplic_list[1][i]
                            count = count + 1
            if (
                spc_mismatching
                == "Caution! Potential differences found between frequency and single-point geometries -"
            ):
                log.write(
                    "\no  No potential differences found between frequency and single-point geometries (based on input coordinates)."
                )
            else:
                spc_mismatching_1 = spc_mismatching[:84]
                spc_mismatching_2 = spc_mismatching[85:]
                log.write("\nx  " + spc_mismatching_1 + spc_mismatching_2 + ".")
        else:
            log.write(
                "\nx  One or more geometries from single-point corrections are missing."
            )

        # Check for SPC dispersion models
        dispersion_check_spc = [
            thermo_data[key].sp_empirical_dispersion for key in thermo_data
        ]
        if all_same(dispersion_check_spc):
            if dispersion_check_spc[0] == "No empirical dispersion detected":
                log.write(
                    "\n-  No empirical dispersion detected in any of the calculations."
                )
            else:
                log.write(
                    "\no  Using "
                    + dispersion_check_spc[0]
                    + " in all the singe-point calculations."
                )
        else:
            print_check_fails(
                log, dispersion_check_spc, file_check, "dispersion models"
            )
        log.write("\n" + STARS + "\n")


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
        v_free = (
            8
            * (
                (1e27 / (solv_molarity * constants.AVOGADRO_CONSTANT)) ** 0.333333
                - solv_volume ** 0.333333
            )
            ** 3
        )
        freespace = v_free * solv_molarity * constants.AVOGADRO_CONSTANT * 1e-24
    else:
        freespace = 1000.0
    return freespace
