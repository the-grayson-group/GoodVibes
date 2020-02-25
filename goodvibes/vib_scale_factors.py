#!/usr/bin/python

# Comments and/or additions are welcome (send e-mail to:
# robert.paton@chem.ox.ac.uk

#####################################
#        vib_scale_factors.py       #
#####################################
###  Written by:  Rob Paton #########
###          and Guilian Luchini  ###
###  Last modified:  2019         ###
#####################################

from collections import namedtuple
import numpy as np

import sys

if sys.version_info > (2,):
    # Py3
    Str_char = "U%d"
else:
    # Py2
    Str_char = "S%d"

"""
Frequency scaling factors, taken from version 4 of The Truhlar group database (https://t1.chem.umn.edu/freqscale/index.html
I. M. Alecu, J. Zheng, Y. Zhao, and D. G. Truhlar, J. Chem. Theory Comput. 6, 2872-2887 (2010).

The array is ordered as:
[level/basis set, zpe_fac, zpe_ref, zpe_meth, harm_fac, harm_ref, harm_meth, fund_fac, fund_ref, fund_meth]
where zpe_fac, harm_fac and fund_fac are the scaling factors for ZPEs, harmonic frequencies, and fundamentals, respectively.

The ref and meth elements refer to original references and method of determinantion by the Truhlar group. All information taken from
https://comp.chem.umn.edu/freqscale/190107_Database_of_Freq_Scale_Factors_v4.pdf

Methods
D: The scale factor was directly obtained from the ZPVE15/10 or F38/10 databases given in Ref. 1.
C: The scale factor was obtained by applying a small systematic correction of -0.0025 to preexisting scale factor. The references for the preexisting (uncorrected) scale factors are given in Supporting Information of Ref. 1 and in Version 1 of this database
R: The scale factor was obtained via the Reduced Scale Factor Optimization Model described in Ref. 1. Briefly, this entails using the ZPE6 database for determining ZPE scale factors, and/or using the universal scale factor ratios of aF/ZPE = 0.974 and aH/ZPE = 1.014 to obtain the respective values for the scale factors for fundamental and harmonic frequencies.
"""
scaling_refs = np.array(
    [
        ("none"),
        (
            "I. M. Alecu, J. Zheng, Y. Zhao, and D. G. Truhlar, J. Chem. Theory Comput. 6, 2872-2887 (2010)."
        ),
        (
            "Y. Zhao and D. G. Truhlar, unpublished (2003), modified by systematic correction of -0.0025 by I. M. Alecu (2010)."
        ),
        ("I. M. Alecu, unpublished (2011)."),
        (
            "J. Zheng, R. J. Rocha, M. Pelegrini, L. F. A. Ferrao, E. F. V. Carvalho, O. Roberto-Neto, F. B. C. Machado, and D. G. Truhlar, J. Chem. Phys. 136, 184310/1-10 (2012)."
        ),
        ("J. Zheng and D. G. Truhlar, unpublished (2014)."),
        ("J. Bao and D. G. Truhlar, unpublished (2014)."),
        ("H. Yu, J. Zheng, and D. G. Truhlar, unpublished (2015)"),
        (
            "S. Kanchanakungwankul, J. L. Bao, J. Zheng, I. M. Alecu, B. J. Lynch, Y. Zhao, and D. G. Truhlar, unpublished (2018)"
        ),
    ]
)

scaling_data = np.array(
    [
        ("AM1", 0.948, 1, "R", 0.961, 1, "R", 0.923, 1, "R"),
        ("B1B95/6-31+G(d,p)", 0.971, 1, "C", 0.985, 1, "R", 0.946, 1, "R"),
        ("B1B95/MG3S", 0.973, 1, "C", 0.987, 1, "R", 0.948, 1, "R"),
        ("B1LYP/MG3S", 0.978, 1, "D", 0.994, 1, "D", 0.955, 1, "D"),
        ("B3LYP/6-31G(2df,2p)", 0.981, 1, "C", 0.995, 1, "R", 0.955, 1, "R"),
        ("B3LYP/6-31G(d)", 0.977, 1, "R", 0.991, 1, "R", 0.952, 1, "R"),
        ("B3LYP/aug-cc-pVTZ", 0.985, 3, "R", 0.999, 3, "R", 0.959, 3, "R"),
        ("B3LYP/def2TZVP", 0.985, 3, "R", 0.999, 3, "R", 0.959, 3, "R"),
        ("B3LYP/ma-TZVP", 0.986, 1, "R", 1.0, 1, "R", 0.96, 1, "R"),
        ("B3LYP/MG3S", 0.983, 1, "D", 0.998, 1, "D", 0.96, 1, "D"),
        ("B3P86/6-31G(d)", 0.971, 1, "R", 0.985, 1, "R", 0.946, 1, "R"),
        ("B3PW91/6-31G(d)", 0.972, 1, "R", 0.986, 1, "R", 0.947, 1, "R"),
        ("B973/def2TZVP", 0.974, 8, "D", 0.988, 8, "R", 0.949, 8, "R"),
        ("B973/ma-TZVP", 0.975, 1, "R", 0.989, 1, "R", 0.95, 1, "R"),
        ("B973/MG3S", 0.972, 1, "D", 0.986, 1, "D", 0.947, 1, "D"),
        ("B98/def2TZVP", 0.984, 1, "R", 0.998, 1, "R", 0.958, 1, "R"),
        ("B98/ma-TZVP", 0.985, 1, "R", 0.999, 1, "R", 0.959, 1, "R"),
        ("B98/MG3S", 0.982, 1, "D", 0.995, 1, "D", 0.956, 1, "D"),
        ("BB1K/6-31+G(d,p)", 0.954, 1, "C", 0.967, 1, "R", 0.929, 1, "R"),
        ("BB1K/MG3S", 0.957, 1, "C", 0.97, 1, "R", 0.932, 1, "R"),
        ("BB95/6-31+G(d,p)", 1.011, 1, "C", 1.025, 1, "R", 0.985, 1, "R"),
        ("BB95/MG3S", 1.012, 1, "C", 1.026, 1, "R", 0.986, 1, "R"),
        ("BLYP/6-311G(df,p)", 1.013, 1, "R", 1.027, 1, "R", 0.987, 1, "R"),
        ("BLYP/6-31G(d)", 1.009, 1, "R", 1.023, 1, "R", 0.983, 1, "R"),
        ("BLYP/MG3S", 1.013, 1, "D", 1.031, 1, "D", 0.991, 1, "D"),
        ("BMC-CCSD", 0.985, 1, "D", 1.001, 1, "D", 0.962, 1, "D"),
        ("BMK/ma-TZVP", 0.972, 1, "R", 0.986, 1, "R", 0.947, 1, "R"),
        ("BMK/MG3S", 0.971, 1, "D", 0.984, 1, "D", 0.945, 1, "D"),
        ("BP86/6-31G(d)", 1.007, 1, "R", 1.021, 1, "R", 0.981, 1, "R"),
        ("BP86/ma-TZVP", 1.014, 1, "R", 1.028, 1, "R", 0.988, 1, "R"),
        ("BPW60/6-311+G(d,p)", 0.934, 2, "C", 0.91, 2, "R", 0.947, 2, "R"),
        ("BPW63/MG3S", 0.923, 2, "C", 0.899, 2, "R", 0.936, 2, "R"),
        ("CAM-B3LYP/ma-TZVP", 0.976, 1, "R", 0.99, 1, "R", 0.951, 1, "R"),
        ("CCSD(T)/jul-cc-pVTZ", 0.984, 1, "R", 0.998, 1, "R", 0.958, 1, "R"),
        ("CCSD(T)/aug-cc-pVTZ", 0.987, 1, "R", 1.001, 1, "R", 0.961, 1, "R"),
        ("CCSD(T)-F12/jul-cc-pVTZ", 0.981, 1, "R", 0.995, 1, "R", 0.955, 1, "R"),
        ("CCSD(T)-F12a/cc-pVDZ-F12", 0.983, 11, "R", 0.997, 11, "R", 0.957, 11, "R"),
        ("CCSD(T)-F12a/cc-pVTZ-F12", 0.984, 1, "R", 0.998, 1, "R", 0.958, 1, "R"),
        (
            "CCSD(T)-F12b/VQZF12//CCSD(T)-F12a/TZF",
            0.984,
            13,
            "R",
            0.998,
            13,
            "R",
            0.958,
            13,
            "R",
        ),
        (
            "CCSD(T)-F12b/VQZF12//CCSD(T)-F12a/DZF",
            0.983,
            13,
            "R",
            0.997,
            13,
            "R",
            0.957,
            13,
            "R",
        ),
        ("CCSD/jul-cc-pVTZ", 0.973, 1, "R", 0.987, 1, "R", 0.948, 1, "R"),
        ("CCSD-F12/jul-cc-pVTZ", 0.971, 1, "R", 0.985, 1, "R", 0.946, 1, "R"),
        ("G96LYP80/6-311+G(d,p)", 0.911, 2, "C", 0.887, 2, "R", 0.924, 2, "R"),
        ("G96LYP82/MG3S", 0.907, 2, "C", 0.883, 2, "R", 0.92, 2, "R"),
        ("GAM/def2TZVP", 0.98, 7, "D", 0.994, 7, "D", 0.955, 7, "D"),
        ("GAM/ma-TZVP", 0.981, 7, "D", 0.995, 7, "D", 0.956, 7, "D"),
        ("HF/3-21G", 0.919, 1, "R", 0.932, 1, "R", 0.895, 1, "R"),
        ("HF/6-31+G(d)", 0.911, 1, "R", 0.924, 1, "R", 0.887, 1, "R"),
        ("HF/6-31+G(d,p)", 0.915, 1, "C", 0.928, 1, "R", 0.891, 1, "R"),
        ("HF/6-311G(d,p)", 0.92, 1, "R", 0.933, 1, "R", 0.896, 1, "R"),
        ("HF/6-311G(df,p)", 0.92, 1, "R", 0.933, 1, "R", 0.896, 1, "R"),
        ("HF/6-31G(d)", 0.909, 1, "R", 0.922, 1, "R", 0.885, 1, "R"),
        ("HF/6-31G(d,p)", 0.913, 1, "R", 0.926, 1, "R", 0.889, 1, "R"),
        ("HF/MG3S", 0.919, 1, "D", 0.932, 1, "D", 0.895, 1, "D"),
        ("HFLYP/MG3S", 0.899, 1, "D", 0.912, 1, "D", 0.876, 1, "D"),
        ("HSEh1PBE/ma-TZVP", 0.979, 1, "R", 0.993, 1, "R", 0.954, 1, "R"),
        ("M05/aug-cc-pVTZ", 0.978, 1, "R", 0.992, 1, "R", 0.953, 1, "R"),
        ("M05/def2TZVP", 0.978, 3, "R", 0.991, 3, "R", 0.952, 3, "R"),
        ("M05/ma-TZVP", 0.979, 1, "R", 0.993, 1, "R", 0.954, 1, "R"),
        ("M05/maug-cc-pVTZ", 0.978, 1, "R", 0.992, 1, "R", 0.953, 1, "R"),
        ("M05/MG3S", 0.977, 1, "D", 0.989, 1, "D", 0.951, 1, "D"),
        ("M052X/6-31+G(d,p)", 0.961, 1, "D", 0.974, 1, "D", 0.936, 1, "D"),
        ("M052X/aug-cc-pVTZ", 0.964, 1, "R", 0.977, 1, "R", 0.939, 1, "R"),
        ("M052X/def2TZVPP", 0.962, 1, "D", 0.976, 1, "D", 0.938, 1, "D"),
        ("M052X/ma-TZVP", 0.965, 1, "R", 0.979, 1, "R", 0.94, 1, "R"),
        ("M052X/maug-cc-pVTZ", 0.964, 1, "R", 0.977, 1, "R", 0.939, 1, "R"),
        ("M052X/MG3S", 0.962, 1, "D", 0.975, 1, "D", 0.937, 1, "D"),
        ("M06/6-31+G(d,p)", 0.98, 1, "D", 0.989, 1, "D", 0.95, 1, "D"),
        ("M06/aug-cc-pVTZ", 0.984, 1, "R", 0.998, 1, "R", 0.958, 1, "R"),
        ("M06/def2TZVP", 0.982, 3, "R", 0.996, 3, "R", 0.956, 3, "R"),
        ("M06/def2TZVPP", 0.979, 1, "D", 0.992, 1, "D", 0.953, 1, "D"),
        ("M06/ma-TZVP", 0.982, 1, "R", 0.996, 1, "R", 0.956, 1, "R"),
        ("M06/maug-cc-pVTZ", 0.982, 1, "R", 0.996, 1, "R", 0.956, 1, "R"),
        ("M06/MG3S", 0.981, 1, "D", 0.994, 1, "D", 0.955, 1, "D"),
        ("M062X/6-31+G(d,p)", 0.967, 1, "D", 0.979, 1, "D", 0.94, 1, "D"),
        ("M062X/6-311+G(d,p)", 0.97, 5, "D", 0.983, 5, "R", 0.944, 5, "R"),
        ("M062X/6-311++G(d,p)", 0.97, 5, "D", 0.983, 5, "R", 0.944, 5, "R"),
        ("M062X/aug-cc-pVDZ", 0.979, 14, "D", 0.993, 14, "R", 0.954, 14, "R"),
        ("M062X/aug-cc-pVTZ", 0.971, 1, "D", 0.985, 1, "D", 0.946, 1, "D"),
        ("M062X/def2TZVP", 0.971, 7, "D", 0.984, 7, "D", 0.946, 7, "D"),
        ("M062X/def2QZVP", 0.97, 7, "D", 0.983, 7, "D", 0.945, 7, "D"),
        ("M062X/def2TZVPP", 0.97, 1, "D", 0.983, 1, "D", 0.945, 1, "D"),
        ("M062X/jul-cc-pVDZ", 0.977, 14, "D", 0.991, 14, "R", 0.952, 14, "R"),
        ("M062X/jul-cc-pVTZ", 0.971, 14, "D", 0.985, 14, "R", 0.946, 14, "R"),
        ("M062X/jun-cc-pVDZ", 0.976, 14, "D", 0.99, 14, "R", 0.951, 14, "R"),
        ("M062X/jun-cc-pVTZ", 0.971, 14, "D", 0.985, 14, "R", 0.946, 14, "R"),
        ("M062X/ma-TZVP", 0.972, 1, "R", 0.986, 1, "R", 0.947, 1, "R"),
        ("M062X/maug-cc-pV(T+d)Z", 0.971, 1, "D", 0.984, 1, "D", 0.945, 1, "D"),
        ("M062X/MG3S", 0.97, 1, "D", 0.982, 1, "D", 0.944, 1, "D"),
        ("M06HF/6-31+G(d,p)", 0.954, 1, "D", 0.969, 1, "D", 0.931, 1, "D"),
        ("M06HF/aug-cc-pVTZ", 0.961, 1, "R", 0.974, 1, "R", 0.936, 1, "R"),
        ("M06HF/def2TZVPP", 0.958, 1, "D", 0.97, 1, "D", 0.932, 1, "D"),
        ("M06HF/ma-TZVP", 0.957, 1, "R", 0.97, 1, "R", 0.932, 1, "R"),
        ("M06HF/maug-cc-pVTZ", 0.959, 1, "R", 0.972, 1, "R", 0.934, 1, "R"),
        ("M06HF/MG3S", 0.955, 1, "D", 0.967, 1, "D", 0.93, 1, "D"),
        ("M06L/6-31G(d,p)", 0.977, 15, "D", 0.991, 15, "R", 0.952, 15, "R"),
        ("M06L/6-31+G(d,p)", 0.978, 1, "D", 0.992, 1, "D", 0.953, 1, "D"),
        ("M06L/aug-cc-pVTZ", 0.98, 1, "R", 0.994, 1, "R", 0.955, 1, "R"),
        ("M06L/aug-cc-pV(T+d)Z", 0.98, 9, "R", 0.994, 9, "R", 0.955, 9, "R"),
        ("M06L/aug-cc-pVTZ-pp", 0.98, 9, "R", 0.994, 9, "R", 0.955, 9, "R"),
        ("M06L(DKH2)/aug-cc-pwcVTZ-DK", 0.985, 1, "D", 0.999, 1, "R", 0.959, 1, "R"),
        ("M06L/def2TZVP", 0.976, 3, "R", 0.99, 3, "R", 0.951, 3, "R"),
        ("M06L/def2TZVPP", 0.976, 1, "D", 0.995, 1, "D", 0.956, 1, "D"),
        ("M06L/ma-TZVP", 0.977, 1, "R", 0.991, 1, "R", 0.952, 1, "R"),
        ("M06L/maug-cc-pVTZ", 0.977, 1, "R", 0.991, 1, "R", 0.952, 1, "R"),
        ("M06L/MG3S", 0.978, 1, "D", 0.996, 1, "D", 0.958, 1, "D"),
        ("M08HX/6-31+G(d,p)", 0.972, 1, "D", 0.983, 1, "D", 0.944, 1, "D"),
        ("M08HX/aug-cc-pVTZ", 0.975, 1, "R", 0.989, 1, "R", 0.95, 1, "R"),
        ("M08HX/cc-pVTZ+", 0.974, 1, "D", 0.985, 1, "D", 0.946, 1, "D"),
        ("M08HX/def2TZVPP", 0.973, 1, "D", 0.984, 1, "D", 0.945, 1, "D"),
        ("M08HX/jun-cc-pVTZ", 0.974, 6, "D", 0.986, 6, "D", 0.947, 6, "D"),
        ("M08HX/ma-TZVP", 0.976, 1, "R", 0.99, 1, "R", 0.951, 1, "R"),
        ("M08HX/maug-cc-pVTZ", 0.976, 1, "R", 0.99, 1, "R", 0.951, 1, "R"),
        ("M08HX/MG3S", 0.973, 1, "D", 0.984, 1, "D", 0.946, 1, "D"),
        ("M08SO/6-31+G(d,p)", 0.979, 1, "D", 0.989, 1, "D", 0.951, 1, "D"),
        ("M08SO/aug-cc-pVTZ", 0.985, 1, "R", 0.999, 1, "R", 0.959, 1, "R"),
        ("M08SO/cc-pVTZ+", 0.982, 1, "D", 0.995, 1, "D", 0.956, 1, "D"),
        ("M08SO/def2TZVPP", 0.98, 1, "D", 0.993, 1, "D", 0.954, 1, "D"),
        ("M08SO/ma-TZVP", 0.984, 1, "R", 0.998, 1, "R", 0.958, 1, "R"),
        ("M08SO/maug-cc-pVTZ", 0.983, 1, "R", 0.997, 1, "R", 0.957, 1, "R"),
        ("M08SO/MG3", 0.984, 4, "D", 0.998, 4, "R", 0.959, 4, "R"),
        ("M08SO/MG3S", 0.983, 1, "D", 0.995, 1, "D", 0.956, 1, "D"),
        ("M08SO/MG3SXP", 0.984, 1, "D", 0.996, 1, "D", 0.957, 1, "D"),
        ("M11L/maug-cc-pVTZ", 0.988, 16, "D", 1.002, 16, "R", 0.962, 16, "R"),
        ("MN11-L/MG3S", 0.985, 16, "D", 0.999, 16, "R", 0.959, 16, "R"),
        ("MN12L/jul-cc-pVDZ", 0.974, 14, "R", 0.988, 14, "R", 0.95, 14, "R"),
        ("MN12L/MG3S", 0.968, 6, "D", 0.981, 6, "D", 0.943, 6, "D"),
        ("MN12SX/6-311++G(d,p)", 0.976, 6, "D", 0.986, 6, "D", 0.947, 6, "D"),
        ("MN12SX/jul-cc-pVDZ", 0.979, 14, "R", 0.993, 14, "R", 0.954, 14, "R"),
        ("MN15L/MG3S", 0.977, 1, "D", 0.991, 1, "R", 0.952, 1, "R"),
        ("MN15L/maug-cc-pVTZ", 0.979, 1, "D", 0.993, 1, "R", 0.954, 1, "R"),
        ("MC3BB", 0.965, 1, "C", 0.979, 1, "R", 0.94, 1, "R"),
        ("MC3MPW", 0.964, 1, "C", 0.977, 1, "R", 0.939, 1, "R"),
        ("MC-QCISD/3", 0.992, 1, "C", 1.006, 1, "R", 0.966, 1, "R"),
        ("MOHLYP/ma-TZVP", 1.027, 1, "R", 1.041, 1, "R", 1.0, 1, "R"),
        ("MOHLYP/MG3S", 1.022, 1, "R", 1.036, 1, "R", 0.995, 1, "R"),
        ("MP2(FC)/6-31+G(d,p)", 0.968, 1, "C", 0.982, 1, "R", 0.943, 1, "R"),
        ("MP2(FC)/6-311G(d,p)", 0.97, 1, "R", 0.984, 1, "R", 0.945, 1, "R"),
        ("MP2(FC)/6-31G(d)", 0.964, 1, "R", 0.977, 1, "R", 0.939, 1, "R"),
        ("MP2(FC)/6-31G(d,p)", 0.958, 1, "R", 0.971, 1, "R", 0.933, 1, "R"),
        ("MP2(FC)/cc-pVDZ", 0.977, 1, "C", 0.991, 1, "R", 0.952, 1, "R"),
        ("MP2(FC)/cc-pVTZ", 0.975, 1, "D", 0.992, 1, "D", 0.953, 1, "D"),
        ("MP2(FULL)/6-31G(d)", 0.963, 1, "R", 0.976, 1, "R", 0.938, 1, "R"),
        ("MP4(SDQ)/jul-cc-pVTZ", 0.973, 1, "R", 0.987, 1, "R", 0.948, 1, "R"),
        ("MPW1B95/6-31+G(d,p)", 0.97, 1, "C", 0.984, 1, "R", 0.945, 1, "R"),
        ("MPW1B95/MG3", 0.97, 1, "C", 0.984, 1, "R", 0.945, 1, "R"),
        ("MPW1B95/MG3S", 0.972, 1, "C", 0.986, 1, "R", 0.947, 1, "R"),
        ("MPW1K/6-31+G(d,p)", 0.949, 1, "C", 0.962, 1, "R", 0.924, 1, "R"),
        ("MPW1K/aug-cc-PDTZ", 0.959, 14, "R", 0.972, 14, "R", 0.934, 14, "R"),
        ("MPW1K/aug-cc-PVTZ", 0.955, 14, "R", 0.968, 14, "R", 0.93, 14, "R"),
        ("MPW1K/jul-cc-pVDZ", 0.957, 14, "R", 0.97, 14, "R", 0.932, 14, "R"),
        ("MPW1K/jul-cc-pVTZ", 0.954, 14, "R", 0.967, 14, "R", 0.929, 14, "R"),
        ("MPW1K/jun-cc-pVDZ", 0.955, 14, "R", 0.968, 14, "R", 0.93, 14, "R"),
        ("MPW1K/jun-cc-pVTZ", 0.954, 14, "R", 0.967, 14, "R", 0.929, 14, "R"),
        ("MPW1K/ma-TZVP", 0.956, 1, "R", 0.969, 1, "R", 0.931, 1, "R"),
        ("MPW1K/MG3", 0.953, 1, "C", 0.966, 1, "R", 0.928, 1, "R"),
        ("MPW1K/MG3S", 0.956, 1, "C", 0.969, 1, "R", 0.931, 1, "R"),
        ("MPW1K/MIDI!", 0.953, 1, "R", 0.966, 1, "R", 0.928, 1, "R"),
        ("MPW1K/MIDIY", 0.947, 1, "R", 0.96, 1, "R", 0.922, 1, "R"),
        ("MPW3LYP/6-31+G(d,p)", 0.98, 1, "C", 0.994, 1, "R", 0.955, 1, "R"),
        ("MPW3LYP/6-311+G(2d,p)", 0.986, 1, "R", 1.0, 1, "R", 0.96, 1, "R"),
        ("MPW3LYP/6-31G(d)", 0.976, 1, "R", 0.99, 1, "R", 0.951, 1, "R"),
        ("MPW3LYP/ma-TZVP", 0.986, 1, "R", 1.0, 1, "R", 0.96, 1, "R"),
        ("MPW3LYP/MG3S", 0.982, 1, "C", 0.996, 1, "R", 0.956, 1, "R"),
        ("MPW74/6-311+G(d,p)", 0.912, 2, "C", 0.888, 2, "R", 0.925, 2, "R"),
        ("MPW76/MG3S", 0.909, 2, "C", 0.885, 2, "R", 0.922, 2, "R"),
        ("MPWB1K/6-31+G(d,p)", 0.951, 1, "C", 0.964, 1, "R", 0.926, 1, "R"),
        ("MPWB1K/MG3S", 0.954, 1, "C", 0.967, 1, "R", 0.929, 1, "R"),
        ("MPWLYP1M/ma-TZVP", 1.009, 1, "R", 1.023, 1, "R", 0.983, 1, "R"),
        ("MW3.2//CCSD(T)-F12a/TZF", 0.984, 13, "R", 0.998, 13, "R", 0.958, 13, "R"),
        ("OreLYP/ma-TZVP", 1.01, 7, "D", 1.024, 7, "D", 0.984, 7, "D"),
        ("OreLYP/def2TZVP", 1.008, 7, "D", 1.023, 7, "D", 0.982, 7, "D"),
        ("PBE/def2TZVP", 1.011, 3, "R", 1.026, 3, "R", 0.985, 3, "R"),
        ("PBE/MG3S", 1.01, 1, "D", 1.025, 1, "D", 0.985, 1, "D"),
        ("PBE/ma-TZVP", 1.014, 1, "D", 1.028, 1, "D", 0.987, 1, "D"),
        ("PBE0/MG3S", 0.975, 1, "D", 0.989, 1, "D", 0.95, 1, "D"),
        ("PBE1KCIS/MG3", 0.981, 1, "C", 0.995, 1, "R", 0.955, 1, "R"),
        ("PBE1KCIS/MG3S", 0.981, 1, "C", 0.995, 1, "R", 0.955, 1, "R"),
        ("PM3", 0.94, 1, "R", 0.953, 1, "R", 0.916, 1, "R"),
        ("PM6", 1.078, 1, "R", 1.093, 1, "R", 1.05, 1, "R"),
        ("PW6B95/def2TZVP", 0.974, 8, "R", 0.988, 8, "R", 0.949, 8, "R"),
        ("PWB6K/cc-pVDZ", 0.953, 12, "D", 0.966, 12, "R", 0.928, 12, "R"),
        ("QCISD/cc-pVTZ", 0.975, 11, "R", 0.989, 11, "R", 0.95, 11, "R"),
        ("QCISD/MG3S", 0.978, 10, "R", 0.992, 10, "R", 0.953, 10, "R"),
        ("QCISD(FC)/6-31G(d)", 0.973, 1, "R", 0.987, 1, "R", 0.948, 1, "R"),
        ("QCISD(T)/aug-cc-pVQZ", 0.989, 10, "R", 1.003, 10, "R", 0.963, 10, "R"),
        ("revTPSS/def2TZVP", 0.998, 7, "D", 1.012, 7, "D", 0.972, 7, "D"),
        ("revTPSS/ma-TZVP", 0.999, 7, "D", 1.013, 7, "D", 0.973, 7, "D"),
        ("SOGGA/ma-TZVP", 1.017, 1, "R", 1.031, 1, "R", 0.991, 1, "R"),
        ("THCTHhyb/ma-TZVP", 0.989, 1, "R", 1.003, 1, "R", 0.963, 1, "R"),
        ("TPSS1KCIS/def2TZVP", 0.982, 1, "R", 0.996, 1, "R", 0.956, 1, "R"),
        ("TPSS1KCIS/ma-TZVP", 0.983, 1, "R", 0.997, 1, "R", 0.957, 1, "R"),
        ("TPSSh/MG3S", 0.984, 1, "D", 1.002, 1, "D", 0.963, 1, "D"),
        ("VSXC/MG3S", 0.986, 1, "D", 1.001, 1, "D", 0.962, 1, "D"),
        ("wB97/def2TZVP", 0.969, 1, "R", 0.983, 1, "R", 0.944, 1, "R"),
        ("wB97/ma-TZVP", 0.97, 1, "R", 0.984, 1, "R", 0.945, 1, "R"),
        ("wB97X/def2TZVP", 0.97, 1, "R", 0.984, 1, "R", 0.945, 1, "R"),
        ("wB97X/ma-TZVP", 0.971, 1, "R", 0.985, 1, "R", 0.946, 1, "R"),
        ("wB97XD/def2TZVP", 0.975, 1, "R", 0.989, 1, "R", 0.95, 1, "R"),
        ("wB97XD/ma-TZVP", 0.975, 1, "R", 0.989, 1, "R", 0.95, 1, "R"),
        ("wB97XD/maug-cc-pVTZ", 0.974, 1, "R", 0.988, 1, "R", 0.949, 1, "R"),
        ("W3X//CCSD(T)-F12a/TZF", 0.984, 13, "R", 0.998, 13, "R", 0.958, 13, "R"),
        ("W3XL//CCSD(T)-F12a/TZF", 0.984, 13, "R", 0.998, 13, "R", 0.958, 13, "R"),
        ("W3XL//QCISD/STZ", 0.973, 13, "R", 0.987, 13, "R", 0.948, 13, "R"),
        ("X1B95/6-31+G(d,p)", 0.968, 1, "C", 0.982, 1, "R", 0.943, 1, "R"),
        ("X1B95/MG3S", 0.971, 1, "C", 0.985, 1, "R", 0.946, 1, "R"),
        ("XB1K/6-31+G(d,p)", 0.952, 1, "C", 0.965, 1, "R", 0.927, 1, "R"),
        ("XB1K/MG3S", 0.955, 1, "C", 0.968, 1, "R", 0.93, 1, "R"),
    ],
    dtype=[
        ("level", Str_char % 40),
        ("zpe_fac", "f4"),
        ("zpe_ref", "i4"),
        ("zpe_meth", Str_char % 1),
        ("harm_fac", "f4"),
        ("harm_ref", "i4"),
        ("harm_meth", Str_char % 1),
        ("fund_fac", "f4"),
        ("fund_ref", "i4"),
        ("fund_meth", Str_char % 1),
    ],
)

ScalingData = namedtuple(
    "ScalingData",
    [
        "level_basis",
        "zpe_fac",
        "zpe_ref",
        "zpe_meth",
        "harm_fac",
        "harm_ref",
        "harm_meth",
        "fund_fac",
        "fund_ref",
        "fund_meth",
    ],
)

scaling_data_dict, scaling_data_dict_mod = {}, {}
for row in scaling_data:
    (
        level_basis,
        zpe_fac,
        zpe_ref,
        zpe_meth,
        harm_fac,
        harm_ref,
        harm_meth,
        fund_fac,
        fund_ref,
        fund_meth,
    ) = row
    data = ScalingData(*row)
    scaling_data_dict[level_basis.upper()] = scaling_data_dict_mod[
        level_basis.replace("-", "").upper()
    ] = data

