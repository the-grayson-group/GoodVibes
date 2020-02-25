import math
from . import constants, utils


# Translational entropy evaluation
# Depends on mass, concentration, temperature, solvent free space: default = 1000.0
def calc_translational_entropy(molecular_mass, conc, temperature, solv):
    """
    Calculates the translational entropic contribution (J/(mol*K)) of an ideal gas.
    Needs the molecular mass. Convert mass in amu to kg; conc in mol/l to number per m^3
    Strans = R(Ln(2pimkT/h^2)^3/2(1/C)) + 1 + 3/2)
    """
    lmda = (
        (
            2.0
            * math.pi
            * molecular_mass
            * constants.AMU_to_KG
            * constants.BOLTZMANN_CONSTANT
            * temperature
        )
        ** 0.5
    ) / constants.PLANCK_CONSTANT
    freespace = utils.get_free_space(solv)
    ndens = conc * 1000 * constants.AVOGADRO_CONSTANT / (freespace / 1000.0)
    entropy = constants.GAS_CONSTANT * (2.5 + math.log(lmda ** 3 / ndens))
    return entropy


# Electronic entropy evaluation
# Depends on multiplicity
def calc_electronic_entropy(multiplicity):
    """
    Calculates the electronic entropic contribution (J/(mol*K)) of the molecule
    Selec = R(Ln(multiplicity)
    """
    entropy = constants.GAS_CONSTANT * (math.log(multiplicity))
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
            entropy = constants.GAS_CONSTANT * (math.log(qrot / symmno) + 1)
        elif linear == 2:
            entropy = 0.0
        else:
            entropy = constants.GAS_CONSTANT * (math.log(qrot / symmno) + 1.5)
    return entropy


# Rigid rotor harmonic oscillator (RRHO) entropy evaluation - this is the default treatment
def calc_rrho_entropy(frequency_wn, temperature, freq_scale_factor, fract_modelsys):
    """
    Entropic contributions (J/(mol*K)) according to a rigid-rotor
    harmonic-oscillator description for a list of vibrational modes
    Sv = RSum(hv/(kT(e^(hv/kT)-1) - ln(1-e^(-hv/kT)))
    """
    if fract_modelsys is not False:
        freq_scale_factor = [
            freq_scale_factor[0] * fract_modelsys[i]
            + freq_scale_factor[1] * (1.0 - fract_modelsys[i])
            for i in range(len(fract_modelsys))
        ]
        factor = [
            (
                constants.PLANCK_CONSTANT
                * frequency_wn[i]
                * constants.SPEED_OF_LIGHT
                * freq_scale_factor[i]
            )
            / (constants.BOLTZMANN_CONSTANT * temperature)
            for i in range(len(frequency_wn))
        ]
    else:
        factor = [
            (
                constants.PLANCK_CONSTANT
                * freq
                * constants.SPEED_OF_LIGHT
                * freq_scale_factor
            )
            / (constants.BOLTZMANN_CONSTANT * temperature)
            for freq in frequency_wn
        ]
    entropy = [
        entry * constants.GAS_CONSTANT / (math.exp(entry) - 1)
        - constants.GAS_CONSTANT * math.log(1 - math.exp(-entry))
        for entry in factor
    ]
    return entropy


# Free rotor entropy evaluation
# used for low frequencies below the cut-off if qs=grimme is specified
def calc_freerot_entropy(frequency_wn, temperature, freq_scale_factor, fract_modelsys):
    """
    Entropic contributions (J/(mol*K)) according to a free-rotor
    description for a list of vibrational modes
    Sr = R(1/2 + 1/2ln((8pi^3u'kT/h^2))
    """
    # This is the average moment of inertia used by Grimme
    bav = 1.00e-44
    if fract_modelsys is not False:
        freq_scale_factor = [
            freq_scale_factor[0] * fract_modelsys[i]
            + freq_scale_factor[1] * (1.0 - fract_modelsys[i])
            for i in range(len(fract_modelsys))
        ]
        mu = [
            constants.PLANCK_CONSTANT
            / (
                8
                * math.pi ** 2
                * frequency_wn[i]
                * constants.SPEED_OF_LIGHT
                * freq_scale_factor[i]
            )
            for i in range(len(frequency_wn))
        ]
    else:
        mu = [
            constants.PLANCK_CONSTANT
            / (8 * math.pi ** 2 * freq * constants.SPEED_OF_LIGHT * freq_scale_factor)
            for freq in frequency_wn
        ]
    mu_primed = [entry * bav / (entry + bav) for entry in mu]
    factor = [
        8
        * math.pi ** 3
        * entry
        * constants.BOLTZMANN_CONSTANT
        * temperature
        / constants.PLANCK_CONSTANT ** 2
        for entry in mu_primed
    ]
    entropy = [
        (0.5 + math.log(entry ** 0.5)) * constants.GAS_CONSTANT for entry in factor
    ]
    return entropy


# A damping function to interpolate between RRHO and free rotor vibrational entropy values
def calc_damp(frequency_wn, freq_cutoff):
    alpha = 4
    damp = [1 / (1 + (freq_cutoff / entry) ** alpha) for entry in frequency_wn]
    return damp
