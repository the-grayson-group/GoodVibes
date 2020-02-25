from . import constants
import math
import sys


# Translational energy evaluation
# Depends on temperature
def calc_translational_energy(temperature):
    """
    Calculates the translational energy (J/mol) of an ideal gas
    i.e. non-interacting molecules so molar energy = Na * atomic energy.
    This approximation applies to all energies and entropies computed within
    Etrans = 3/2 RT!
    """
    energy = 1.5 * constants.GAS_CONSTANT * temperature
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
        energy = constants.GAS_CONSTANT * temperature
    else:
        energy = 1.5 * constants.GAS_CONSTANT * temperature
    return energy


# Vibrational energy evaluation
# Depends on frequencies, temperature and scaling factor: default = 1.0
def calc_vibrational_energy(
    frequency_wn, temperature, freq_scale_factor, fract_modelsys
):
    """
    Calculates the vibrational energy contribution (J/mol).
    Includes ZPE (0K) and thermal contributions
    Evib = R * Sum(0.5 hv/k + (hv/k)/(e^(hv/KT)-1))
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
    # Error occurs if T is too low when performing math.exp
    for entry in factor:
        if entry > math.log(sys.float_info.max):
            sys.exit(
                "\nx  Warning! Temperature may be too low to calculate vibrational energy. Please adjust using the `-t` option and try again.\n"
            )

    energy = [
        entry
        * constants.GAS_CONSTANT
        * temperature
        * (0.5 + (1.0 / (math.exp(entry) - 1.0)))
        for entry in factor
    ]

    return sum(energy)


# Vibrational Zero point energy evaluation
# Depends on frequencies and scaling factor: default = 1.0
def calc_zeropoint_energy(frequency_wn, freq_scale_factor, fract_modelsys):
    """
    Calculates the vibrational ZPE (J/mol)
    EZPE = Sum(0.5 hv/k)
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
            / (constants.BOLTZMANN_CONSTANT)
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
            / (constants.BOLTZMANN_CONSTANT)
            for freq in frequency_wn
        ]
    energy = [0.5 * entry * constants.GAS_CONSTANT for entry in factor]
    return sum(energy)


# Quasi-rigid rotor harmonic oscillator energy evaluation
# used for calculating quasi-harmonic enthalpy
def calc_qRRHO_energy(frequency_wn, temperature, freq_scale_factor):
    """
    Head-Gordon RRHO-vibrational energy contribution (J/mol*K) of
    vibrational modes described by a rigid-rotor harmonic approximation
    V_RRHO = 1/2(Nhv) + RT(hv/kT)e^(-hv/kT)/(1-e^(-hv/kT))
    """
    factor = [
        constants.PLANCK_CONSTANT * freq * constants.SPEED_OF_LIGHT * freq_scale_factor
        for freq in frequency_wn
    ]
    energy = [
        0.5 * constants.AVOGADRO_CONSTANT * entry
        + constants.GAS_CONSTANT
        * temperature
        * entry
        / constants.BOLTZMANN_CONSTANT
        / temperature
        * math.exp(-entry / constants.BOLTZMANN_CONSTANT / temperature)
        / (1 - math.exp(-entry / constants.BOLTZMANN_CONSTANT / temperature))
        for entry in factor
    ]
    return energy
