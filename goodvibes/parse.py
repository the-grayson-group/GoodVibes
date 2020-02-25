import os
from . import constants


# Read Gaussian output and obtain single point energy, program type,
# program version, solvation_model, charge, empirical_dispersion, multiplicity
def parse_data(file):
    (
        spe,
        program,
        data,
        version_program,
        solvation_model,
        keyword_line,
        a,
        charge,
        multiplicity,
    ) = ("none", "none", [], "", "", "", 0, None, None)

    if os.path.exists(os.path.splitext(file)[0] + ".log"):
        with open(os.path.splitext(file)[0] + ".log") as f:
            data = f.readlines()
    elif os.path.exists(os.path.splitext(file)[0] + ".out"):
        with open(os.path.splitext(file)[0] + ".out") as f:
            data = f.readlines()
    else:
        raise ValueError("File {} does not exist".format(file))

    for line in data:
        if "Gaussian" in line:
            program = "Gaussian"
            break
        if "* O   R   C   A *" in line:
            program = "Orca"
            break
    repeated_link1 = 0
    for line in data:
        if program == "Gaussian":
            if line.strip().startswith("SCF Done:"):
                spe = float(line.strip().split()[4])
            if line.strip().startswith("Counterpoise corrected energy"):
                spe = float(line.strip().split()[4])
            # For MP2 calculations replace with EUMP2
            if "EUMP2 =" in line.strip():
                spe = float((line.strip().split()[5]).replace("D", "E"))
            # For ONIOM calculations use the extrapolated value rather than SCF value
            if "ONIOM: extrapolated energy" in line.strip():
                spe = float(line.strip().split()[4])
            # For Semi-empirical or Molecular Mechanics calculations
            if (
                "Energy= " in line.strip()
                and "Predicted" not in line.strip()
                and "Thermal" not in line.strip()
            ):
                spe = float(line.strip().split()[1])
            if "Gaussian" in line and "Revision" in line and repeated_link1 == 0:
                for i in range(len(line.strip(",").split(",")) - 1):
                    line.strip(",").split(",")[i]
                    version_program += line.strip(",").split(",")[i]
                    repeated_link1 = 1
                version_program = version_program[1:]
            if "Charge" in line.strip() and "Multiplicity" in line.strip():
                charge = line.split("Multiplicity")[0].split("=")[-1].strip()
                multiplicity = line.split("=")[-1].strip()
        if program == "Orca":
            if line.strip().startswith("FINAL SINGLE POINT ENERGY"):
                spe = float(line.strip().split()[4])
            if "Program Version" in line.strip():
                version_program = "ORCA version " + line.split()[2]
            if "Total Charge" in line.strip() and "...." in line.strip():
                charge = int(line.strip("=").split()[-1])
            if "Multiplicity" in line.strip() and "...." in line.strip():
                multiplicity = int(line.strip("=").split()[-1])

    # Solvation model and empirical dispersion detection
    if "Gaussian" in version_program.strip():
        for i, line in enumerate(data):
            if "#" in line.strip() and a == 0:
                for j, line in enumerate(data[i : i + 10]):
                    if "--" in line.strip():
                        a = a + 1
                        break
                    if a != 0:
                        break
                    else:
                        for k in range(len(line.strip().split("\n"))):
                            line.strip().split("\n")[k]
                            keyword_line += line.strip().split("\n")[k]
        keyword_line = keyword_line.lower()
        if "scrf" not in keyword_line.strip():
            solvation_model = "gas phase"
        else:
            start_scrf = keyword_line.strip().find("scrf") + 4
            if "(" in keyword_line[start_scrf : start_scrf + 4]:
                start_scrf += keyword_line[start_scrf : start_scrf + 4].find("(") + 1
                end_scrf = keyword_line.find(")", start_scrf)
                display_solvation_model = (
                    "scrf=("
                    + ",".join(keyword_line[start_scrf:end_scrf].lower().split(","))
                    + ")"
                )
                sorted_solvation_model = (
                    "scrf=("
                    + ",".join(
                        sorted(keyword_line[start_scrf:end_scrf].lower().split(","))
                    )
                    + ")"
                )
            else:
                if " = " in keyword_line[start_scrf : start_scrf + 4]:
                    start_scrf += (
                        keyword_line[start_scrf : start_scrf + 4].find(" = ") + 3
                    )
                elif " =" in keyword_line[start_scrf : start_scrf + 4]:
                    start_scrf += (
                        keyword_line[start_scrf : start_scrf + 4].find(" =") + 2
                    )
                elif "=" in keyword_line[start_scrf : start_scrf + 4]:
                    start_scrf += (
                        keyword_line[start_scrf : start_scrf + 4].find("=") + 1
                    )
                end_scrf = keyword_line.find(" ", start_scrf)
                if end_scrf == -1:
                    display_solvation_model = (
                        "scrf=("
                        + ",".join(keyword_line[start_scrf:].lower().split(","))
                        + ")"
                    )
                    sorted_solvation_model = (
                        "scrf=("
                        + ",".join(sorted(keyword_line[start_scrf:].lower().split(",")))
                        + ")"
                    )
                else:
                    display_solvation_model = (
                        "scrf=("
                        + ",".join(keyword_line[start_scrf:end_scrf].lower().split(","))
                        + ")"
                    )
                    sorted_solvation_model = (
                        "scrf=("
                        + ",".join(
                            sorted(keyword_line[start_scrf:end_scrf].lower().split(","))
                        )
                        + ")"
                    )
        if solvation_model != "gas phase":
            solvation_model = [sorted_solvation_model, display_solvation_model]
        empirical_dispersion = ""
        if (
            keyword_line.strip().find("empiricaldispersion") == -1
            and keyword_line.strip().find("emp=") == -1
            and keyword_line.strip().find("emp =") == -1
            and keyword_line.strip().find("emp(") == -1
        ):
            empirical_dispersion = "No empirical dispersion detected"
        elif keyword_line.strip().find("empiricaldispersion") > -1:
            start_emp_disp = keyword_line.strip().find("empiricaldispersion") + 19
            if "(" in keyword_line[start_emp_disp : start_emp_disp + 4]:
                start_emp_disp += (
                    keyword_line[start_emp_disp : start_emp_disp + 4].find("(") + 1
                )
                end_emp_disp = keyword_line.find(")", start_emp_disp)
                empirical_dispersion = (
                    "empiricaldispersion=("
                    + ",".join(
                        sorted(
                            keyword_line[start_emp_disp:end_emp_disp].lower().split(",")
                        )
                    )
                    + ")"
                )
            else:
                if " = " in keyword_line[start_emp_disp : start_emp_disp + 4]:
                    start_emp_disp += (
                        keyword_line[start_emp_disp : start_emp_disp + 4].find(" = ")
                        + 3
                    )
                elif " =" in keyword_line[start_emp_disp : start_emp_disp + 4]:
                    start_emp_disp += (
                        keyword_line[start_emp_disp : start_emp_disp + 4].find(" =") + 2
                    )
                elif "=" in keyword_line[start_emp_disp : start_emp_disp + 4]:
                    start_emp_disp += (
                        keyword_line[start_emp_disp : start_emp_disp + 4].find("=") + 1
                    )
                end_emp_disp = keyword_line.find(" ", start_emp_disp)
                if end_emp_disp == -1:
                    empirical_dispersion = (
                        "empiricaldispersion=("
                        + ",".join(
                            sorted(keyword_line[start_emp_disp:].lower().split(","))
                        )
                        + ")"
                    )
                else:
                    empirical_dispersion = (
                        "empiricaldispersion=("
                        + ",".join(
                            sorted(
                                keyword_line[start_emp_disp:end_emp_disp]
                                .lower()
                                .split(",")
                            )
                        )
                        + ")"
                    )
        elif (
            keyword_line.strip().find("emp=") > -1
            or keyword_line.strip().find("emp =") > -1
            or keyword_line.strip().find("emp(") > -1
        ):
            # Check for temp keyword
            temp, emp_e, emp_p = False, False, False
            check_temp = keyword_line.strip().find("emp=")
            start_emp_disp = keyword_line.strip().find("emp=")
            if check_temp == -1:
                check_temp = keyword_line.strip().find("emp =")
                start_emp_disp = keyword_line.strip().find("emp =")
            if check_temp == -1:
                check_temp = keyword_line.strip().find("emp=(")
                start_emp_disp = keyword_line.strip().find("emp(")
            check_temp += -1
            if keyword_line[check_temp].lower() == "t":
                temp = True  # Look for a new one
                if keyword_line.strip().find("emp=", check_temp + 5) > -1:
                    emp_e = True
                    start_emp_disp = (
                        keyword_line.strip().find("emp=", check_temp + 5) + 3
                    )
                elif keyword_line.strip().find("emp =", check_temp + 5) > -1:
                    emp_e = True
                    start_emp_disp = (
                        keyword_line.strip().find("emp =", check_temp + 5) + 3
                    )
                elif keyword_line.strip().find("emp(", check_temp + 5) > -1:
                    emp_p = True
                    start_emp_disp = (
                        keyword_line.strip().find("emp(", check_temp + 5) + 3
                    )
                else:
                    empirical_dispersion = "No empirical dispersion detected"
            else:
                start_emp_disp += 3
            if (
                (temp and emp_e)
                or (not temp and keyword_line.strip().find("emp=") > -1)
                or (not temp and keyword_line.strip().find("emp ="))
            ):
                if "(" in keyword_line[start_emp_disp : start_emp_disp + 4]:
                    start_emp_disp += (
                        keyword_line[start_emp_disp : start_emp_disp + 4].find("(") + 1
                    )
                    end_emp_disp = keyword_line.find(")", start_emp_disp)
                    empirical_dispersion = (
                        "empiricaldispersion=("
                        + ",".join(
                            sorted(
                                keyword_line[start_emp_disp:end_emp_disp]
                                .lower()
                                .split(",")
                            )
                        )
                        + ")"
                    )
                else:
                    if " = " in keyword_line[start_emp_disp : start_emp_disp + 4]:
                        start_emp_disp += (
                            keyword_line[start_emp_disp : start_emp_disp + 4].find(
                                " = "
                            )
                            + 3
                        )
                    elif " =" in keyword_line[start_emp_disp : start_emp_disp + 4]:
                        start_emp_disp += (
                            keyword_line[start_emp_disp : start_emp_disp + 4].find(" =")
                            + 2
                        )
                    elif "=" in keyword_line[start_emp_disp : start_emp_disp + 4]:
                        start_emp_disp += (
                            keyword_line[start_emp_disp : start_emp_disp + 4].find("=")
                            + 1
                        )
                    end_emp_disp = keyword_line.find(" ", start_emp_disp)
                    if end_emp_disp == -1:
                        empirical_dispersion = (
                            "empiricaldispersion=("
                            + ",".join(
                                sorted(keyword_line[start_emp_disp:].lower().split(","))
                            )
                            + ")"
                        )
                    else:
                        empirical_dispersion = (
                            "empiricaldispersion=("
                            + ",".join(
                                sorted(
                                    keyword_line[start_emp_disp:end_emp_disp]
                                    .lower()
                                    .split(",")
                                )
                            )
                            + ")"
                        )
            elif (temp and emp_p) or (
                not temp and keyword_line.strip().find("emp(") > -1
            ):
                start_emp_disp += (
                    keyword_line[start_emp_disp : start_emp_disp + 4].find("(") + 1
                )
                end_emp_disp = keyword_line.find(")", start_emp_disp)
                empirical_dispersion = (
                    "empiricaldispersion=("
                    + ",".join(
                        sorted(
                            keyword_line[start_emp_disp:end_emp_disp].lower().split(",")
                        )
                    )
                    + ")"
                )
    if "ORCA" in version_program.strip():
        keyword_line_1 = "gas phase"
        keyword_line_2 = ""
        keyword_line_3 = ""
        for i, line in enumerate(data):
            if "CPCM SOLVATION MODEL" in line.strip():
                keyword_line_1 = "CPCM,"
            if "SMD CDS free energy correction energy" in line.strip():
                keyword_line_2 = "SMD,"
            if "Solvent:              " in line.strip():
                keyword_line_3 = line.strip().split()[-1]
        solvation_model = keyword_line_1 + keyword_line_2 + keyword_line_3
        empirical_dispersion1 = "No empirical dispersion detected"
        empirical_dispersion2 = ""
        empirical_dispersion3 = ""
        for i, line in enumerate(data):
            if keyword_line.strip().find("DFT DISPERSION CORRECTION") > -1:
                empirical_dispersion1 = ""
            if keyword_line.strip().find("DFTD3") > -1:
                empirical_dispersion2 = "D3"
            if keyword_line.strip().find("USING zero damping") > -1:
                empirical_dispersion3 = " with zero damping"
        empirical_dispersion = (
            empirical_dispersion1 + empirical_dispersion2 + empirical_dispersion3
        )
    return (
        spe,
        program,
        version_program,
        solvation_model,
        file,
        charge,
        empirical_dispersion,
        multiplicity,
    )


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
                if (
                    line.find("(" + name.split(".")[0] + ")") > -1
                    and line.find("Compound") > -1
                ):
                    if data[i - 5].find("Temperature") > -1:
                        temp = data[i - 5].split()[2]
                    if float(temp) > float(interval[0]) and float(temp) < float(
                        interval[1]
                    ):
                        if float(temp) not in t_interval:
                            t_interval.append(float(temp))
                        if data[i + 10].find("Gibbs") > -1:
                            gsolv = (
                                float(data[i + 10].split()[6].strip())
                                / constants.KCAL_TO_AU
                            )
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
                if (
                    line.find("(" + name.split(".")[0] + ")") > -1
                    and line.find("Compound") > -1
                ):
                    if data[i + 11].find("Gibbs") > -1:
                        gsolv = (
                            float(data[i + 11].split()[6].strip())
                            / constants.KCAL_TO_AU
                        )
                        gsolv[name] = gsolv

    if interval:
        return t_interval, gsolv_dicts
    else:
        return gsolv


# Read output for the level of theory and basis set used
def level_of_theory(file):
    repeated_theory = 0
    with open(file) as f:
        data = f.readlines()
    level, bs = "none", "none"

    for line in data:
        if line.strip().find("External calculation") > -1:
            level, bs = "ext", "ext"
            break
        if "\\Freq\\" in line.strip() and repeated_theory == 0:
            try:
                level, bs = line.strip().split("\\")[4:6]
                repeated_theory = 1
            except IndexError:
                pass
        elif "|Freq|" in line.strip() and repeated_theory == 0:
            try:
                level, bs = line.strip().split("|")[4:6]
                repeated_theory = 1
            except IndexError:
                pass
        if "\\SP\\" in line.strip() and repeated_theory == 0:
            try:
                level, bs = line.strip().split("\\")[4:6]
                repeated_theory = 1
            except IndexError:
                pass
        elif "|SP|" in line.strip() and repeated_theory == 0:
            try:
                level, bs = line.strip().split("|")[4:6]
                repeated_theory = 1
            except IndexError:
                pass
        if "DLPNO BASED TRIPLES CORRECTION" in line.strip():
            level = "DLPNO-CCSD(T)"
        if "Estimated CBS total energy" in line.strip():
            try:
                bs = "Extrapol." + line.strip().split()[4]
            except IndexError:
                pass
        # Remove the restricted R or unrestricted U label
        if level[0] in ("R", "U"):
            level = level[1:]
    level_of_theory = "/".join([level, bs])
    return level_of_theory


# Read output for the level of theory and basis set used
def jobtype(file):
    with open(file) as f:
        data = f.readlines()
    job = ""
    for line in data:
        if line.strip().find("\\SP\\") > -1:
            job += "SP"
        if line.strip().find("\\FOpt\\") > -1:
            job += "GS"
        if line.strip().find("\\FTS\\") > -1:
            job += "TS"
        if line.strip().find("\\Freq\\") > -1:
            job += "Freq"
    return job


# Read single-point output for cpu time
def sp_cpu(file):
    spe, program, data, cpu = None, None, [], None

    if os.path.exists(os.path.splitext(file)[0] + ".log"):
        with open(os.path.splitext(file)[0] + ".log") as f:
            data = f.readlines()
    elif os.path.exists(os.path.splitext(file)[0] + ".out"):
        with open(os.path.splitext(file)[0] + ".out") as f:
            data = f.readlines()
    else:
        raise ValueError("File {} does not exist".format(file))

    for line in data:
        if line.find("Gaussian") > -1:
            program = "Gaussian"
            break
        if line.find("* O   R   C   A *") > -1:
            program = "Orca"
            break

    for line in data:
        if program == "Gaussian":
            if line.strip().startswith("SCF Done:"):
                spe = float(line.strip().split()[4])
            if line.strip().find("Job cpu time") > -1:
                days = int(line.split()[3])
                hours = int(line.split()[5])
                mins = int(line.split()[7])
                secs = 0
                msecs = int(float(line.split()[9]) * 1000.0)
                cpu = [days, hours, mins, secs, msecs]
        if program == "Orca":
            if line.strip().startswith("FINAL SINGLE POINT ENERGY"):
                spe = float(line.strip().split()[4])
            if line.strip().find("TOTAL RUN TIME") > -1:
                days = int(line.split()[3])
                hours = int(line.split()[5])
                mins = int(line.split()[7])
                secs = int(line.split()[9])
                msecs = float(line.split()[11])
                cpu = [days, hours, mins, secs, msecs]

    return cpu


# At beginning of procedure, read level of theory, solvation model, and check for normal termination
def read_initial(file):
    with open(file) as f:
        data = f.readlines()
    level, bs, program, keyword_line = "none", "none", "none", "none"
    progress, orientation = "Incomplete", "Input"
    a, repeated_theory = 0, 0
    no_grid = True
    DFT, dft_used, level, bs, scf_iradan, cphf_iradan = (
        False,
        "F",
        "none",
        "none",
        False,
        False,
    )
    grid_lookup = {1: "sg1", 2: "coarse", 4: "fine", 5: "ultrafine", 7: "superfine"}

    for line in data:
        # Determine program to find solvation model used
        if "Gaussian" in line:
            program = "Gaussian"
        if "* O   R   C   A *" in line:
            program = "Orca"
        # Grab pertinent information from file
        if line.strip().find("External calculation") > -1:
            level, bs = "ext", "ext"
        if line.strip().find("Standard orientation:") > -1:
            orientation = "Standard"
        if line.strip().find("IExCor=") > -1 and no_grid:
            try:
                dft_used = line.split("=")[2].split()[0]
                grid = grid_lookup[int(dft_used)]
                no_grid = False
            except:
                pass
        if "\\Freq\\" in line.strip() and repeated_theory == 0:
            try:
                level, bs = line.strip().split("\\")[4:6]
                repeated_theory = 1
            except IndexError:
                pass
        elif "|Freq|" in line.strip() and repeated_theory == 0:
            try:
                level, bs = line.strip().split("|")[4:6]
                repeated_theory = 1
            except IndexError:
                pass
        if "\\SP\\" in line.strip() and repeated_theory == 0:
            try:
                level, bs = line.strip().split("\\")[4:6]
                repeated_theory = 1
            except IndexError:
                pass
        elif "|SP|" in line.strip() and repeated_theory == 0:
            try:
                level, bs = line.strip().split("|")[4:6]
                repeated_theory = 1
            except IndexError:
                pass
        if "DLPNO BASED TRIPLES CORRECTION" in line.strip():
            level = "DLPNO-CCSD(T)"
        if "Estimated CBS total energy" in line.strip():
            try:
                bs = "Extrapol." + line.strip().split()[4]
            except IndexError:
                pass
        # Remove the restricted R or unrestricted U label
        if level[0] in ("R", "U"):
            level = level[1:]

    # Grab solvation models - Gaussian files
    if program is "Gaussian":
        for i, line in enumerate(data):
            if "#" in line.strip() and a == 0:
                for j, line in enumerate(data[i : i + 10]):
                    if "--" in line.strip():
                        a = a + 1
                        break
                    if a != 0:
                        break
                    else:
                        for k in range(len(line.strip().split("\n"))):
                            line.strip().split("\n")[k]
                            keyword_line += line.strip().split("\n")[k]
            if "Normal termination" in line:
                progress = "Normal"
            elif "Error termination" in line:
                progress = "Error"
        keyword_line = keyword_line.lower()
        if "scrf" not in keyword_line.strip():
            solvation_model = "gas phase"
        else:
            start_scrf = keyword_line.strip().find("scrf") + 5
            if keyword_line[start_scrf] == "(":
                end_scrf = keyword_line.find(")", start_scrf)
                solvation_model = "scrf=" + keyword_line[start_scrf:end_scrf]
                if solvation_model[-1] != ")":
                    solvation_model = solvation_model + ")"
            else:
                start_scrf2 = keyword_line.strip().find("scrf") + 4
                if keyword_line.find(" ", start_scrf) > -1:
                    end_scrf = keyword_line.find(" ", start_scrf)
                else:
                    end_scrf = len(keyword_line)
                if keyword_line[start_scrf2] == "(":
                    solvation_model = "scrf=(" + keyword_line[start_scrf:end_scrf]
                    if solvation_model[-1] != ")":
                        solvation_model = solvation_model + ")"
                else:
                    if keyword_line.find(" ", start_scrf) > -1:
                        end_scrf = keyword_line.find(" ", start_scrf)
                    else:
                        end_scrf = len(keyword_line)
                    solvation_model = "scrf=" + keyword_line[start_scrf:end_scrf]
    # ORCA parsing for solvation model
    elif program is "Orca":
        keyword_line_1 = "gas phase"
        keyword_line_2 = ""
        keyword_line_3 = ""
        for i, line in enumerate(data):
            if "CPCM SOLVATION MODEL" in line.strip():
                keyword_line_1 = "CPCM,"
            if "SMD CDS free energy correction energy" in line.strip():
                keyword_line_2 = "SMD,"
            if "Solvent:              " in line.strip():
                keyword_line_3 = line.strip().split()[-1]
            if "ORCA TERMINATED NORMALLY" in line:
                progress = "Normal"
            elif "error termination" in line:
                progress = "Error"
        solvation_model = keyword_line_1 + keyword_line_2 + keyword_line_3
    level_of_theory = "/".join([level, bs])

    return level_of_theory, solvation_model, progress, orientation, dft_used
