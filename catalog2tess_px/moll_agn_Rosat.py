from astroquery.vizier import Vizier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

Vizier.ROW_LIMIT = -1
rosat_catalog = Vizier.get_catalogs('IX/30A')[0]
rosat_ra = np.array(rosat_catalog['RAJ2000'].data, dtype=float)
rosat_dec = np.array(rosat_catalog['DEJ2000'].data, dtype=float)

with open('List.txt', 'r') as f:
    target_objects = [line.strip() for line in f.readlines()]

light_curve_files = glob.glob('processed_light_curves*.txt')
matched_ra = []
matched_dec = []

columns = ['Name', 'Objtype', 'Agnclass', 'RA', 'DEC', 'Mean_Flux', 'Stddev', 
          'Sector', 'Camera', 'CCD', 'Chi2_Normalized', 'Chi2_reduced_Normalized',
          'Chi2_Standardized', 'Chi2_reduced_Standardized']

for lcfile in light_curve_files:
    df = pd.read_csv(lcfile, delim_whitespace=True, names=columns)
    matched_objects = df[df['Name'].isin(target_objects)]
    matched_ra.extend(matched_objects['RA'].tolist())
    matched_dec.extend(matched_objects['DEC'].tolist())

matched_ra = np.array(matched_ra, dtype=float)
matched_dec = np.array(matched_dec, dtype=float)

plt.figure(figsize=(16, 10))
ax = plt.subplot(111, projection='mollweide')
ax.scatter(np.radians(rosat_ra), np.radians(rosat_dec), 
          color='gray', s=1, alpha=0.3, label='ROSAT Sources')
ax.scatter(np.radians(matched_ra), np.radians(matched_dec), 
          color='red', s=50, alpha=0.8, label='Target Objects')

plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='upper right')
plt.title('ROSAT Sources vs Target Objects Distribution', fontsize=16)

plt.savefig('combined_sky_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Plotted {len(rosat_ra)} ROSAT sources and {len(matched_ra)} target objects")
