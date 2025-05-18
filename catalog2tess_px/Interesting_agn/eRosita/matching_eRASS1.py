from astropy.io import fits
import numpy as np
import pandas as pd
import glob

with fits.open('eRASS1_clusters_optical.fits') as hdul:
    erass_data = hdul[1].data

results = []
sector_files = glob.glob('sector*_interesting.txt')

for file in sector_files:
    data = pd.read_csv(file, delim_whitespace=True)

    for idx, row in data.iterrows():
        ra_deg = row['RA'] * 15.0 
        
        ra_diff = np.abs(erass_data['RA'] - ra_deg)
        dec_diff = np.abs(erass_data['DEC'] - row['DEC'])
        matches = (ra_diff < (10/3600)) & (dec_diff < (10/3600))
        
        obj_data = {
            'Name': row['Name'],
            'RA': ra_deg,
            'DEC': row['DEC'],
            'BEST_Z': None,
            'LIT_Z': None
        }

        if np.any(matches):
            match_idx = np.where(matches)[0][0]
            obj_data.update({
                'BEST_Z': erass_data['BEST_Z'][match_idx] if 'BEST_Z' in erass_data.dtype.names else None,
                'LIT_Z': erass_data['LIT_Z'][match_idx] if 'LIT_Z' in erass_data.dtype.names else None
            })

        results.append(obj_data)

df = pd.DataFrame(results)

with open('erass1_redshifts.txt', 'w') as f:
    header = ['Name', 'RA', 'DEC', 'BEST_Z', 'LIT_Z']
    f.write('\t'.join(header) + '\n')

    for _, row in df.iterrows():
        line = [str(row[col]) if pd.notnull(row[col]) else 'NULL' for col in header]
        f.write('\t'.join(line) + '\n')

total_objects = len(results)
found_objects = len(df.dropna(subset=['ML_FLUX_0'], how='all'))
print(f"Found eRASS1 matches for {found_objects} out of {total_objects} objects")
print("Results saved to erass1_matches.txt")
