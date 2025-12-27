#!/usr/bin/env python3
"""
ANM B-factor Prediction and Output Tool - Command Line Version
"""

from prody import *
import numpy as np
import pandas as pd
import argparse
import os
import sys

# Residue name conversion dictionary
THREE_TO_ONE = {
    'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
    'GLN':'Q', 'GLU':'E', 'GLY':'G', 'HIS':'H', 'ILE':'I',
    'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
    'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'
}

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='ANM B-factor prediction and CSV export')
    
    parser.add_argument('-e', '--experimental-pdb', required=True,
                        help='Path to experimental structure PDB file')
    parser.add_argument('-d', '--design-pdb', required=True,
                        help='Path to design structure PDB file')
    parser.add_argument('-o', '--output', default='b_factor_report.csv',
                        help='Output CSV file name (default: b_factor_report.csv)')
    parser.add_argument('-m', '--modes', type=int, nargs=2, default=[6, 20],
                        metavar=('START', 'END'),
                        help='Range of ANM modes to use (default: 6 20)')
    parser.add_argument('-x', '--exclude-tail', type=int, default=388,
                        help='Number of C-terminal residues to exclude (default: 388)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed output')

    return parser.parse_args()

def validate_inputs(args):
    """Validate input files"""
    if not os.path.exists(args.experimental_pdb):
        sys.exit(f"Error: Experimental PDB file does not exist {args.experimental_pdb}")
    if not os.path.exists(args.design_pdb):
        sys.exit(f"Error: Design PDB file does not exist {args.design_pdb}")
    
    if args.modes[0] >= args.modes[1]:
        sys.exit("Error: Mode range must be ascending (start < end)")
    if args.exclude_tail < 0:
        sys.exit("Error: Number of residues to exclude cannot be negative")

def main():
    # Parse arguments
    args = parse_arguments()
    validate_inputs(args)

    # Load structure files
    try:
        if args.debug: print("Loading experimental structure...")
        exp_structure = parsePDB(args.experimental_pdb)
        exp_calphas = exp_structure.select("calpha")
        
        if args.debug: print("Loading design structure...")
        des_structure = parsePDB(args.design_pdb)
        des_calphas = des_structure.select('calpha')
    except Exception as e:
        sys.exit(f"Failed to load PDB files: {str(e)}")

    # Calculate ANM modes
    try:
        if args.debug: print(f"Calculating ANM modes {args.modes[0]}-{args.modes[1]}...")
        anm = ANM('ANM analysis')
        anm.buildHessian(des_calphas)
        anm.calcModes()
        
        # Adjust mode indices (user input starts at 1, Python starts at 0)
        mode_start = args.modes[0] - 1
        mode_end = args.modes[1]
        pred_bfactors = calcTempFactors(anm[mode_start:mode_end], des_calphas)
    except IndexError:
        sys.exit("Error: Requested mode range exceeds available modes")
    except Exception as e:
        sys.exit(f"ANM calculation failed: {str(e)}")

    # Normalization
    exp_bfactors = exp_calphas.getBetas()
    exp_min, exp_max = np.min(exp_bfactors), np.max(exp_bfactors)
    
    scaled_bfactors = (pred_bfactors - np.min(pred_bfactors)) / \
                     (np.max(pred_bfactors) - np.min(pred_bfactors)) * \
                     (exp_max - exp_min) + exp_min

    # Residue slicing
    n_res = len(des_calphas)
    keep_num = n_res - args.exclude_tail
    if keep_num <= 0:
        sys.exit(f"Error: Number of residues to exclude ({args.exclude_tail}) exceeds total residues ({n_res})")
    
    des_calphas = des_calphas[:keep_num]
    scaled_bfactors = scaled_bfactors[:keep_num]

    # Build data table
    data = []
    for atom, b_factor in zip(des_calphas, scaled_bfactors):
        resname_3 = atom.getResname()
        data.append({
            "Chain": atom.getChid().strip(),
            "ResID": atom.getResnum(),
            "ResName": THREE_TO_ONE.get(resname_3, 'X'),
            "B_factor": round(b_factor, 2)
        })

    # Save CSV
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    
    print(f"Report successfully generated: {args.output}")
    print(f"Residues included: {keep_num}/{n_res}")
    print(f"ANM modes used: {args.modes[0]}-{args.modes[1]}")
    print(f"Experimental B-factor range: {exp_min:.2f}-{exp_max:.2f}")

if __name__ == "__main__":
    main()
