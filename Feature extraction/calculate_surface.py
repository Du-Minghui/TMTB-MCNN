#!/usr/bin/env python3
"""
Command-line tool to calculate protein surface residue probability based on DSSP file
Usage: python dssp_surface_prob.py -i input.dssp -o output.csv --chain A
"""

import argparse
import pandas as pd
from pathlib import Path

# Theoretical maximum SASA values (Å²)
MAX_SASA = {
    'A': 117.5,   # Alanine
    'R': 257.3,   # Arginine
    'N': 176.1,   # Asparagine
    'D': 172.2,   # Aspartate
    'C': 150.3,   # Cysteine
    'E': 201.3,   # Glutamate
    'Q': 204.6,   # Glutamine
    'G': 93.5,    # Glycine
    'H': 209.1,   # Histidine
    'I': 188.8,   # Isoleucine
    'L': 191.3,   # Leucine
    'K': 225.7,   # Lysine
    'M': 208.6,   # Methionine
    'F': 227.2,   # Phenylalanine
    'P': 150.7,   # Proline
    'S': 137.2,   # Serine
    'T': 158.3,   # Threonine
    'W': 268.3,   # Tryptophan
    'Y': 245.9,   # Tyrosine
    'V': 165.9    # Valine
}

def calc_surface_prob(res_aa, sasa):
    """Calculate surface probability based on maximum SASA"""
    if res_aa not in MAX_SASA or sasa is None:
        return None
    rsa = sasa / MAX_SASA[res_aa]
    return max(0.0, min(1.0, rsa))

def parse_dssp(dssp_file, chain=None):
    """
    Parse DSSP file and extract residue information
    :param dssp_file: Input DSSP file path
    :param chain: Extract only specified chain (default: all)
    :return: DataFrame containing residue information
    """
    residues = []
    
    with open(dssp_file, 'r') as f:
        # Locate the start of data lines
        while not f.readline().startswith('  #  RESIDUE'):
            continue
        
        for line in f:
            if not line.strip():
                continue
            
            # Extract fields (based on DSSP fixed column positions)
            current_chain = line[11].strip()
            res_id = line[5:11].strip()
            aa = line[13].strip()
            dssp_code = line[16].strip()
            sasa = line[35:39].strip()
            
            # Chain filter
            if chain and current_chain != chain:
                continue
            
            residues.append({
                "Chain": current_chain,
                "ResID": res_id,
                "ResName": aa,
                "DSSP": dssp_code,
                "SASA": float(sasa)
            })
    
    return pd.DataFrame(residues)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculate protein surface residue probability')
    parser.add_argument('-i', '--input', required=True, help='Input DSSP file path')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    parser.add_argument('-c', '--chain', help='Process only specified chain (e.g., A)')
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    
    # Parse DSSP and calculate probabilities
    df = parse_dssp(args.input, chain=args.chain)
    df['Surface_Prob'] = df.apply(
        lambda row: calc_surface_prob(row['ResName'], row['SASA']), 
        axis=1
    ).round(4)
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"✅ Results saved to {args.output}, processed {len(df)} residues")

if __name__ == "__main__":
    main()
