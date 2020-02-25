import os
import sys
import math
from . import constants


# Obtain relative thermochemistry between species and for reactions
class get_pes:
    def __init__(
        self, file, thermo_data, log, temperature, gconf, QH, cosmo=None, cosmo_int=None
    ):
        # Default values
        self.dec, self.units, self.boltz = 2, "kcal/mol", False

        with open(file) as f:
            data = f.readlines()
        folder, program, names, files, zeros, pes_list = None, None, [], [], [], []
        for i, dline in enumerate(data):
            if dline.strip().find("PES") > -1:
                for j, line in enumerate(data[i + 1 :]):
                    if line.strip().startswith("#"):
                        pass
                    elif len(line) <= 2:
                        pass
                    elif line.strip().startswith("---"):
                        break
                    elif line.strip() != "":
                        pathway, pes = line.strip().replace(":", "=").split("=")
                        # Auto-grab first species as zero unless specified
                        pes_list.append(pes)
                        zeros.append(pes.strip().lstrip("[").rstrip("]").split(",")[0])
                        # Look at SPECIES block to determine filenames
            if dline.strip().find("SPECIES") > -1:
                for j, line in enumerate(data[i + 1 :]):
                    if line.strip().startswith("---"):
                        break
                    else:
                        if line.lower().strip().find("folder") > -1:
                            try:
                                folder = (
                                    line.strip().replace("#", "=").split("=")[1].strip()
                                )
                            except IndexError:
                                pass
                        else:
                            try:
                                n, f = line.strip().replace(":", "=").split("=")
                                # Check the specified filename is also one that GoodVibes has thermochemistry for:
                                if f.find("*") == -1 and f not in pes_list:
                                    match = None
                                    for key in thermo_data:
                                        if os.path.splitext(os.path.basename(key))[
                                            0
                                        ] in f.replace("[", "").replace(
                                            "]", ""
                                        ).replace(
                                            "+", ","
                                        ).replace(
                                            " ", ""
                                        ).split(
                                            ","
                                        ):
                                            match = key
                                    if match:
                                        names.append(n.strip())
                                        files.append(match)
                                    else:
                                        log.write(
                                            "   Warning! "
                                            + f.strip()
                                            + " is specified in "
                                            + file
                                            + " but no thermochemistry data found\n"
                                        )
                                elif f not in pes_list:
                                    match = []
                                    for key in thermo_data:
                                        if (
                                            os.path.splitext(os.path.basename(key))[
                                                0
                                            ].find(f.strip().strip("*"))
                                            == 0
                                        ):
                                            match.append(key)
                                    if len(match) > 0:
                                        names.append(n.strip())
                                        files.append(match)
                                    else:
                                        log.write(
                                            "   Warning! "
                                            + f.strip()
                                            + " is specified in "
                                            + file
                                            + " but no thermochemistry data found\n"
                                        )
                            except ValueError:
                                if line.isspace():
                                    pass
                                elif line.strip().find("#") > -1:
                                    pass
                                elif len(line) > 2:
                                    warn = (
                                        "   Warning! "
                                        + file
                                        + " input is incorrectly formatted for line:\n\t"
                                        + line
                                    )
                                    log.write(warn)
            # Look at FORMAT block to see if user has specified any formatting rules
            if dline.strip().find("FORMAT") > -1:
                for j, line in enumerate(data[i + 1 :]):
                    if line.strip().find("dec") > -1:
                        try:
                            self.dec = int(
                                line.strip().replace(":", "=").split("=")[1].strip()
                            )
                        except IndexError:
                            pass
                    if line.strip().find("zero") > -1:
                        zeros = []
                        try:
                            zeros.append(
                                line.strip().replace(":", "=").split("=")[1].strip()
                            )
                        except IndexError:
                            pass
                    if line.strip().find("units") > -1:
                        try:
                            self.units = (
                                line.strip().replace(":", "=").split("=")[1].strip()
                            )
                        except IndexError:
                            pass
                    if line.strip().find("boltz") > -1:
                        try:
                            self.boltz = (
                                line.strip().replace(":", "=").split("=")[1].strip()
                            )
                        except IndexError:
                            pass

        for i in range(len(files)):
            if len(files[i]) is 1:
                files[i] = files[i][0]
        species = dict(zip(names, files))
        self.path, self.species = [], []
        (
            self.spc_abs,
            self.e_abs,
            self.zpe_abs,
            self.h_abs,
            self.qh_abs,
            self.s_abs,
            self.qs_abs,
            self.g_abs,
            self.qhg_abs,
            self.cosmo_qhg_abs,
        ) = ([], [], [], [], [], [], [], [], [], [])
        (
            self.spc_zero,
            self.e_zero,
            self.zpe_zero,
            self.h_zero,
            self.qh_zero,
            self.ts_zero,
            self.qhts_zero,
            self.g_zero,
            self.qhg_zero,
            self.cosmo_qhg_zero,
        ) = ([], [], [], [], [], [], [], [], [], [])
        self.g_qhgvals, self.g_species_qhgzero, self.g_rel_val = [], [], []
        # Loop over .yaml file, grab energies, populate arrays and compute Boltzmann factors
        with open(file) as f:
            data = f.readlines()
        for i, dline in enumerate(data):
            if dline.strip().find("PES") > -1:
                n = 0
                for j, line in enumerate(data[i + 1 :]):
                    if line.strip().startswith("#") == True:
                        pass
                    elif len(line) <= 2:
                        pass
                    elif line.strip().startswith("---") == True:
                        break
                    elif line.strip() != "":
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
                            self.cosmo_qhg_zero.append([])
                            min_conf = False
                            (
                                spc_zero,
                                e_zero,
                                zpe_zero,
                                h_zero,
                                qh_zero,
                                s_zero,
                                qs_zero,
                                g_zero,
                                qhg_zero,
                            ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                            (
                                h_conf,
                                h_tot,
                                s_conf,
                                s_tot,
                                qh_conf,
                                qh_tot,
                                qs_conf,
                                qs_tot,
                                cosmo_qhg_zero,
                            ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                            zero_structures = zeros[n].replace(" ", "").split("+")
                            # Routine for 'zero' values
                            for structure in zero_structures:
                                try:
                                    if not isinstance(species[structure], list):
                                        if hasattr(
                                            thermo_data[species[structure]], "sp_energy"
                                        ):
                                            spc_zero += thermo_data[
                                                species[structure]
                                            ].sp_energy
                                        e_zero += thermo_data[
                                            species[structure]
                                        ].scf_energy
                                        zpe_zero += thermo_data[species[structure]].zpe
                                        h_zero += thermo_data[
                                            species[structure]
                                        ].enthalpy
                                        qh_zero += thermo_data[
                                            species[structure]
                                        ].qh_enthalpy
                                        s_zero += thermo_data[
                                            species[structure]
                                        ].entropy
                                        qs_zero += thermo_data[
                                            species[structure]
                                        ].qh_entropy
                                        g_zero += thermo_data[
                                            species[structure]
                                        ].gibbs_free_energy
                                        qhg_zero += thermo_data[
                                            species[structure]
                                        ].qh_gibbs_free_energy
                                        cosmo_qhg_zero += thermo_data[
                                            species[structure]
                                        ].cosmo_qhg
                                    else:  # If we have a list of different kinds of structures: loop over conformers
                                        g_min, boltz_sum = sys.float_info.max, 0.0
                                        for conformer in species[
                                            structure
                                        ]:  # Find minimum G, along with associated enthalpy and entropy
                                            if cosmo:
                                                if (
                                                    thermo_data[conformer].cosmo_qhg
                                                    <= g_min
                                                ):
                                                    min_conf = thermo_data[conformer]
                                                    g_min = thermo_data[
                                                        conformer
                                                    ].cosmo_qhg
                                            else:
                                                if (
                                                    thermo_data[
                                                        conformer
                                                    ].qh_gibbs_free_energy
                                                    <= g_min
                                                ):
                                                    min_conf = thermo_data[conformer]
                                                    g_min = thermo_data[
                                                        conformer
                                                    ].qh_gibbs_free_energy
                                        for conformer in species[
                                            structure
                                        ]:  # Get a Boltzmann sum for conformers
                                            if cosmo:
                                                g_rel = (
                                                    thermo_data[conformer].cosmo_qhg
                                                    - g_min
                                                )
                                            else:
                                                g_rel = (
                                                    thermo_data[
                                                        conformer
                                                    ].qh_gibbs_free_energy
                                                    - g_min
                                                )
                                            boltz_fac = math.exp(
                                                -g_rel
                                                * constants.J_TO_AU
                                                / constants.GAS_CONSTANT
                                                / temperature
                                            )
                                            boltz_sum += boltz_fac
                                        for conformer in species[
                                            structure
                                        ]:  # Calculate relative data based on Gmin and the Boltzmann sum
                                            if cosmo:
                                                g_rel = (
                                                    thermo_data[conformer].cosmo_qhg
                                                    - g_min
                                                )
                                            else:
                                                g_rel = (
                                                    thermo_data[
                                                        conformer
                                                    ].qh_gibbs_free_energy
                                                    - g_min
                                                )
                                            boltz_fac = math.exp(
                                                -g_rel
                                                * constants.J_TO_AU
                                                / constants.GAS_CONSTANT
                                                / temperature
                                            )
                                            boltz_prob = boltz_fac / boltz_sum
                                            # if no contribution, skip further calculations
                                            if boltz_prob == 0.0:
                                                continue

                                            if (
                                                hasattr(
                                                    thermo_data[conformer], "sp_energy"
                                                )
                                                and thermo_data[conformer].sp_energy
                                                is not "!"
                                            ):
                                                spc_zero += (
                                                    thermo_data[conformer].sp_energy
                                                    * boltz_prob
                                                )
                                            if (
                                                hasattr(
                                                    thermo_data[conformer], "sp_energy"
                                                )
                                                and thermo_data[conformer].sp_energy
                                                is "!"
                                            ):
                                                sys.exit(
                                                    "Not all files contain a SPC value, relative values will not be calculated."
                                                )
                                            e_zero += (
                                                thermo_data[conformer].scf_energy
                                                * boltz_prob
                                            )
                                            zpe_zero += (
                                                thermo_data[conformer].zpe * boltz_prob
                                            )
                                            # Default calculate gconf correction for conformers, skip if no contribution
                                            if (
                                                gconf
                                                and boltz_prob > 0.0
                                                and boltz_prob != 1.0
                                            ):
                                                h_conf += (
                                                    thermo_data[conformer].enthalpy
                                                    * boltz_prob
                                                )
                                                s_conf += (
                                                    thermo_data[conformer].entropy
                                                    * boltz_prob
                                                )
                                                s_conf += (
                                                    -constants.GAS_CONSTANT
                                                    / constants.J_TO_AU
                                                    * boltz_prob
                                                    * math.log(boltz_prob)
                                                )

                                                qh_conf += (
                                                    thermo_data[conformer].qh_enthalpy
                                                    * boltz_prob
                                                )
                                                qs_conf += (
                                                    thermo_data[conformer].qh_entropy
                                                    * boltz_prob
                                                )
                                                qs_conf += (
                                                    -constants.GAS_CONSTANT
                                                    / constants.J_TO_AU
                                                    * boltz_prob
                                                    * math.log(boltz_prob)
                                                )
                                            elif gconf and boltz_prob == 1.0:
                                                h_conf += thermo_data[
                                                    conformer
                                                ].enthalpy
                                                s_conf += thermo_data[conformer].entropy
                                                qh_conf += thermo_data[
                                                    conformer
                                                ].qh_enthalpy
                                                qs_conf += thermo_data[
                                                    conformer
                                                ].qh_entropy
                                            else:
                                                h_zero += (
                                                    thermo_data[conformer].enthalpy
                                                    * boltz_prob
                                                )
                                                s_zero += (
                                                    thermo_data[conformer].entropy
                                                    * boltz_prob
                                                )
                                                g_zero += (
                                                    thermo_data[
                                                        conformer
                                                    ].gibbs_free_energy
                                                    * boltz_prob
                                                )

                                                qh_zero += (
                                                    thermo_data[conformer].qh_enthalpy
                                                    * boltz_prob
                                                )
                                                qs_zero += (
                                                    thermo_data[conformer].qh_entropy
                                                    * boltz_prob
                                                )
                                                qhg_zero += (
                                                    thermo_data[
                                                        conformer
                                                    ].qh_gibbs_free_energy
                                                    * boltz_prob
                                                )
                                                cosmo_qhg_zero += (
                                                    thermo_data[conformer].cosmo_qhg
                                                    * boltz_prob
                                                )

                                        if gconf:
                                            h_adj = h_conf - min_conf.enthalpy
                                            h_tot = min_conf.enthalpy + h_adj
                                            s_adj = s_conf - min_conf.entropy
                                            s_tot = min_conf.entropy + s_adj
                                            g_corr = h_tot - temperature * s_tot
                                            qh_adj = qh_conf - min_conf.qh_enthalpy
                                            qh_tot = min_conf.qh_enthalpy + qh_adj
                                            qs_adj = qs_conf - min_conf.qh_entropy
                                            qs_tot = min_conf.qh_entropy + qs_adj
                                            if QH:
                                                qg_corr = qh_tot - temperature * qs_tot
                                            else:
                                                qg_corr = h_tot - temperature * qs_tot
                                except KeyError:
                                    log.write(
                                        "   Warning! Structure "
                                        + structure
                                        + " has not been defined correctly as energy-zero in "
                                        + file
                                        + "\n"
                                    )
                                    log.write(
                                        "   Make sure this structure matches one of the SPECIES defined in the same file\n"
                                    )
                                    sys.exit(
                                        "   Please edit " + file + " and try again\n"
                                    )
                            # Set zero vals here
                            conformers, single_structure, mix = False, False, False
                            for structure in zero_structures:
                                if not isinstance(species[structure], list):
                                    single_structure = True
                                else:
                                    conformers = True
                            if conformers and single_structure:
                                mix = True
                            if gconf and min_conf is not False:
                                if mix:
                                    h_mix = h_tot + h_zero
                                    s_mix = s_tot + s_zero
                                    g_mix = g_corr + g_zero
                                    qh_mix = qh_tot + qh_zero
                                    qs_mix = qs_tot + qs_zero
                                    qg_mix = qg_corr + qhg_zero
                                    cosmo_qhg_mix = qg_corr + cosmo_qhg_zero
                                    self.h_zero[n].append(h_mix)
                                    self.ts_zero[n].append(s_mix)
                                    self.g_zero[n].append(g_mix)
                                    self.qh_zero[n].append(qh_mix)
                                    self.qhts_zero[n].append(qs_mix)
                                    self.qhg_zero[n].append(qg_mix)
                                    self.cosmo_qhg_zero[n].append(cosmo_qhg_mix)
                                elif conformers:
                                    self.h_zero[n].append(h_tot)
                                    self.ts_zero[n].append(s_tot)
                                    self.g_zero[n].append(g_corr)
                                    self.qh_zero[n].append(qh_tot)
                                    self.qhts_zero[n].append(qs_tot)
                                    self.qhg_zero[n].append(qg_corr)
                                    self.cosmo_qhg_zero[n].append(qg_corr)
                            else:
                                self.h_zero[n].append(h_zero)
                                self.ts_zero[n].append(s_zero)
                                self.g_zero[n].append(g_zero)

                                self.qh_zero[n].append(qh_zero)
                                self.qhts_zero[n].append(qs_zero)
                                self.qhg_zero[n].append(qhg_zero)
                                self.cosmo_qhg_zero[n].append(cosmo_qhg_zero)

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
                            self.cosmo_qhg_abs.append([])
                            self.g_qhgvals.append([])
                            self.g_species_qhgzero.append([])
                            self.g_rel_val.append([])  # graphing

                            pathway, pes = line.strip().replace(":", "=").split("=")
                            pes = pes.strip()
                            points = [
                                entry.strip()
                                for entry in pes.lstrip("[").rstrip("]").split(",")
                            ]
                            self.path.append(pathway.strip())
                            # Obtain relative values for each species
                            for i, point in enumerate(points):
                                if point != "":
                                    # Create values to populate
                                    point_structures = point.replace(" ", "").split("+")
                                    (
                                        e_abs,
                                        spc_abs,
                                        zpe_abs,
                                        h_abs,
                                        qh_abs,
                                        s_abs,
                                        g_abs,
                                        qs_abs,
                                        qhg_abs,
                                        cosmo_qhg_abs,
                                    ) = (
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                    )
                                    (
                                        qh_conf,
                                        qh_tot,
                                        qs_conf,
                                        qs_tot,
                                        h_conf,
                                        h_tot,
                                        s_conf,
                                        s_tot,
                                        g_corr,
                                        qg_corr,
                                    ) = (
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                    )
                                    min_conf = False
                                    rel_val = 0.0
                                    self.g_qhgvals[n].append([])
                                    self.g_species_qhgzero[n].append([])
                                    try:
                                        for j, structure in enumerate(
                                            point_structures
                                        ):  # Loop over structures, structures are species specified
                                            zero_conf = 0.0
                                            self.g_qhgvals[n][i].append([])
                                            if not isinstance(
                                                species[structure], list
                                            ):  # Only one conf in structures
                                                e_abs += thermo_data[
                                                    species[structure]
                                                ].scf_energy
                                                if hasattr(
                                                    thermo_data[species[structure]],
                                                    "sp_energy",
                                                ):
                                                    spc_abs += thermo_data[
                                                        species[structure]
                                                    ].sp_energy
                                                zpe_abs += thermo_data[
                                                    species[structure]
                                                ].zpe
                                                h_abs += thermo_data[
                                                    species[structure]
                                                ].enthalpy
                                                qh_abs += thermo_data[
                                                    species[structure]
                                                ].qh_enthalpy
                                                s_abs += thermo_data[
                                                    species[structure]
                                                ].entropy
                                                g_abs += thermo_data[
                                                    species[structure]
                                                ].gibbs_free_energy
                                                qs_abs += thermo_data[
                                                    species[structure]
                                                ].qh_entropy
                                                qhg_abs += thermo_data[
                                                    species[structure]
                                                ].qh_gibbs_free_energy
                                                cosmo_qhg_abs += thermo_data[
                                                    species[structure]
                                                ].cosmo_qhg
                                                zero_conf += thermo_data[
                                                    species[structure]
                                                ].qh_gibbs_free_energy
                                                self.g_qhgvals[n][i][j].append(
                                                    thermo_data[
                                                        species[structure]
                                                    ].qh_gibbs_free_energy
                                                )
                                                rel_val += thermo_data[
                                                    species[structure]
                                                ].qh_gibbs_free_energy
                                            else:  # If we have a list of different kinds of structures: loop over conformers
                                                g_min, boltz_sum = (
                                                    sys.float_info.max,
                                                    0.0,
                                                )
                                                # Find minimum G, along with associated enthalpy and entropy
                                                for conformer in species[structure]:
                                                    if cosmo:
                                                        if (
                                                            thermo_data[
                                                                conformer
                                                            ].cosmo_qhg
                                                            <= g_min
                                                        ):
                                                            min_conf = thermo_data[
                                                                conformer
                                                            ]
                                                            g_min = thermo_data[
                                                                conformer
                                                            ].cosmo_qhg
                                                    else:
                                                        if (
                                                            thermo_data[
                                                                conformer
                                                            ].qh_gibbs_free_energy
                                                            <= g_min
                                                        ):
                                                            min_conf = thermo_data[
                                                                conformer
                                                            ]
                                                            g_min = thermo_data[
                                                                conformer
                                                            ].qh_gibbs_free_energy
                                                # Get a Boltzmann sum for conformers
                                                for conformer in species[structure]:
                                                    if cosmo:
                                                        g_rel = (
                                                            thermo_data[
                                                                conformer
                                                            ].cosmo_qhg
                                                            - g_min
                                                        )
                                                    else:
                                                        g_rel = (
                                                            thermo_data[
                                                                conformer
                                                            ].qh_gibbs_free_energy
                                                            - g_min
                                                        )
                                                    boltz_fac = math.exp(
                                                        -g_rel
                                                        * constants.J_TO_AU
                                                        / constants.GAS_CONSTANT
                                                        / temperature
                                                    )
                                                    boltz_sum += boltz_fac
                                                # Calculate relative data based on Gmin and the Boltzmann sum
                                                for conformer in species[structure]:
                                                    if cosmo:
                                                        g_rel = (
                                                            thermo_data[
                                                                conformer
                                                            ].cosmo_qhg
                                                            - g_min
                                                        )
                                                    else:
                                                        g_rel = (
                                                            thermo_data[
                                                                conformer
                                                            ].qh_gibbs_free_energy
                                                            - g_min
                                                        )
                                                    boltz_fac = math.exp(
                                                        -g_rel
                                                        * constants.J_TO_AU
                                                        / constants.GAS_CONSTANT
                                                        / temperature
                                                    )
                                                    boltz_prob = boltz_fac / boltz_sum
                                                    if boltz_prob == 0.0:
                                                        continue
                                                    if (
                                                        hasattr(
                                                            thermo_data[conformer],
                                                            "sp_energy",
                                                        )
                                                        and thermo_data[
                                                            conformer
                                                        ].sp_energy
                                                        is not "!"
                                                    ):
                                                        spc_abs += (
                                                            thermo_data[
                                                                conformer
                                                            ].sp_energy
                                                            * boltz_prob
                                                        )
                                                    if (
                                                        hasattr(
                                                            thermo_data[conformer],
                                                            "sp_energy",
                                                        )
                                                        and thermo_data[
                                                            conformer
                                                        ].sp_energy
                                                        is "!"
                                                    ):
                                                        sys.exit(
                                                            "\n   Not all files contain a SPC value, relative values will not be calculated.\n"
                                                        )
                                                    e_abs += (
                                                        thermo_data[
                                                            conformer
                                                        ].scf_energy
                                                        * boltz_prob
                                                    )
                                                    zpe_abs += (
                                                        thermo_data[conformer].zpe
                                                        * boltz_prob
                                                    )
                                                    if cosmo:
                                                        zero_conf += (
                                                            thermo_data[
                                                                conformer
                                                            ].cosmo_qhg
                                                            * boltz_prob
                                                        )
                                                        rel_val += (
                                                            thermo_data[
                                                                conformer
                                                            ].cosmo_qhg
                                                            * boltz_prob
                                                        )
                                                    else:
                                                        zero_conf += (
                                                            thermo_data[
                                                                conformer
                                                            ].qh_gibbs_free_energy
                                                            * boltz_prob
                                                        )
                                                        rel_val += (
                                                            thermo_data[
                                                                conformer
                                                            ].qh_gibbs_free_energy
                                                            * boltz_prob
                                                        )
                                                    # Default calculate gconf correction for conformers, skip if no contribution
                                                    if (
                                                        gconf
                                                        and boltz_prob > 0.0
                                                        and boltz_prob != 1.0
                                                    ):
                                                        h_conf += (
                                                            thermo_data[
                                                                conformer
                                                            ].enthalpy
                                                            * boltz_prob
                                                        )
                                                        s_conf += (
                                                            thermo_data[
                                                                conformer
                                                            ].entropy
                                                            * boltz_prob
                                                        )
                                                        s_conf += (
                                                            -constants.GAS_CONSTANT
                                                            / constants.J_TO_AU
                                                            * boltz_prob
                                                            * math.log(boltz_prob)
                                                        )

                                                        qh_conf += (
                                                            thermo_data[
                                                                conformer
                                                            ].qh_enthalpy
                                                            * boltz_prob
                                                        )
                                                        qs_conf += (
                                                            thermo_data[
                                                                conformer
                                                            ].qh_entropy
                                                            * boltz_prob
                                                        )
                                                        qs_conf += (
                                                            -constants.GAS_CONSTANT
                                                            / constants.J_TO_AU
                                                            * boltz_prob
                                                            * math.log(boltz_prob)
                                                        )
                                                    elif gconf and boltz_prob == 1.0:
                                                        h_conf += thermo_data[
                                                            conformer
                                                        ].enthalpy
                                                        s_conf += thermo_data[
                                                            conformer
                                                        ].entropy
                                                        qh_conf += thermo_data[
                                                            conformer
                                                        ].qh_enthalpy
                                                        qs_conf += thermo_data[
                                                            conformer
                                                        ].qh_entropy
                                                    else:
                                                        h_abs += (
                                                            thermo_data[
                                                                conformer
                                                            ].enthalpy
                                                            * boltz_prob
                                                        )
                                                        s_abs += (
                                                            thermo_data[
                                                                conformer
                                                            ].entropy
                                                            * boltz_prob
                                                        )
                                                        g_abs += (
                                                            thermo_data[
                                                                conformer
                                                            ].gibbs_free_energy
                                                            * boltz_prob
                                                        )

                                                        qh_abs += (
                                                            thermo_data[
                                                                conformer
                                                            ].qh_enthalpy
                                                            * boltz_prob
                                                        )
                                                        qs_abs += (
                                                            thermo_data[
                                                                conformer
                                                            ].qh_entropy
                                                            * boltz_prob
                                                        )
                                                        qhg_abs += (
                                                            thermo_data[
                                                                conformer
                                                            ].qh_gibbs_free_energy
                                                            * boltz_prob
                                                        )
                                                        cosmo_qhg_abs += (
                                                            thermo_data[
                                                                conformer
                                                            ].cosmo_qhg
                                                            * boltz_prob
                                                        )
                                                    if cosmo:
                                                        self.g_qhgvals[n][i][j].append(
                                                            thermo_data[
                                                                conformer
                                                            ].cosmo_qhg
                                                        )
                                                    else:
                                                        self.g_qhgvals[n][i][j].append(
                                                            thermo_data[
                                                                conformer
                                                            ].qh_gibbs_free_energy
                                                        )
                                                if gconf:
                                                    h_adj = h_conf - min_conf.enthalpy
                                                    h_tot = min_conf.enthalpy + h_adj
                                                    s_adj = s_conf - min_conf.entropy
                                                    s_tot = min_conf.entropy + s_adj
                                                    g_corr = h_tot - temperature * s_tot
                                                    qh_adj = (
                                                        qh_conf - min_conf.qh_enthalpy
                                                    )
                                                    qh_tot = (
                                                        min_conf.qh_enthalpy + qh_adj
                                                    )
                                                    qs_adj = (
                                                        qs_conf - min_conf.qh_entropy
                                                    )
                                                    qs_tot = (
                                                        min_conf.qh_entropy + qs_adj
                                                    )
                                                    if QH:
                                                        qg_corr = (
                                                            qh_tot
                                                            - temperature * qs_tot
                                                        )
                                                    else:
                                                        qg_corr = (
                                                            h_tot - temperature * qs_tot
                                                        )
                                            self.g_species_qhgzero[n][i].append(
                                                zero_conf
                                            )  # Raw data for graphing
                                    except KeyError:
                                        log.write(
                                            "   Warning! Structure "
                                            + structure
                                            + " has not been defined correctly in "
                                            + file
                                            + "\n"
                                        )
                                        sys.exit(
                                            "   Please edit "
                                            + file
                                            + " and try again\n"
                                        )
                                    self.species[n].append(point)
                                    self.e_abs[n].append(e_abs)
                                    self.spc_abs[n].append(spc_abs)
                                    self.zpe_abs[n].append(zpe_abs)
                                    conformers, single_structure, mix = (
                                        False,
                                        False,
                                        False,
                                    )
                                    self.g_rel_val[n].append(rel_val)
                                    for structure in point_structures:
                                        if not isinstance(species[structure], list):
                                            single_structure = True
                                        else:
                                            conformers = True
                                    if conformers and single_structure:
                                        mix = True
                                    if gconf and min_conf is not False:
                                        if mix:
                                            h_mix = h_tot + h_abs
                                            s_mix = s_tot + s_abs
                                            g_mix = g_corr + g_abs
                                            qh_mix = qh_tot + qh_abs
                                            qs_mix = qs_tot + qs_abs
                                            qg_mix = qg_corr + qhg_abs
                                            cosmo_qhg_mix = qg_corr + cosmo_qhg_zero
                                            self.h_abs[n].append(h_mix)
                                            self.s_abs[n].append(s_mix)
                                            self.g_abs[n].append(g_mix)
                                            self.qh_abs[n].append(qh_mix)
                                            self.qs_abs[n].append(qs_mix)
                                            self.qhg_abs[n].append(qg_mix)
                                            self.cosmo_qhg_abs[n].append(cosmo_qhg_mix)
                                        elif conformers:
                                            self.h_abs[n].append(h_tot)
                                            self.s_abs[n].append(s_tot)
                                            self.g_abs[n].append(g_corr)
                                            self.qh_abs[n].append(qh_tot)
                                            self.qs_abs[n].append(qs_tot)
                                            self.qhg_abs[n].append(qg_corr)
                                            self.cosmo_qhg_abs[n].append(qg_corr)
                                    else:
                                        self.h_abs[n].append(h_abs)
                                        self.s_abs[n].append(s_abs)
                                        self.g_abs[n].append(g_abs)

                                        self.qh_abs[n].append(qh_abs)
                                        self.qs_abs[n].append(qs_abs)
                                        self.qhg_abs[n].append(qhg_abs)
                                        self.cosmo_qhg_abs[n].append(cosmo_qhg_abs)
                                else:
                                    self.species[n].append("none")
                                    self.e_abs[n].append(float("nan"))

                            n = n + 1
                        except IndexError:
                            pass
