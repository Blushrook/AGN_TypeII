import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import glob

plt.figure(figsize=(15, 10))
ax = plt.subplot(111, projection='mollweide')
sector_files = glob.glob('sector*_interesting.txt')
colors = plt.cm.rainbow(np.linspace(0, 1, len(sector_files)))

for file, color in zip(sector_files, colors):
    data = pd.read_csv(file, delim_whitespace=True)
    ra_degrees = data['RA'] * (180.0/12.0)
    coords = SkyCoord(ra=ra_degrees*u.degree, dec=data['DEC']*u.degree)
    ra_rad = coords.ra.wrap_at(180*u.deg).radian
    dec_rad = coords.dec.radian
    sector_num = file.split('_')[0]
    ax.scatter(ra_rad, dec_rad, s=50, alpha=0.6, c=[color], 
              label=sector_num, marker='o')

ax.grid(True)
plt.title('Mollweide Projection of Sky Coordinates by Sector')
ax.set_xlabel('Right Ascension')
ax.set_ylabel('Declination')

plt.legend(bbox_to_anchor=(1.12, 1), loc='upper right')
plt.savefig('interesting_mollweide.png', dpi=300, bbox_inches='tight')
plt.close()
