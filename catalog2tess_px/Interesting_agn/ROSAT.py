from astroquery.vizier import Vizier
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import glob

Vizier.ROW_LIMIT = -1
rosat_catalog = Vizier.get_catalogs('IX/30A')[0]
rosat_ra = np.array(rosat_catalog['RAJ2000'].data, dtype=float)
rosat_dec = np.array(rosat_catalog['DEJ2000'].data, dtype=float)

rosat_ra_rad = np.radians(rosat_ra)
rosat_ra_rad[rosat_ra_rad > np.pi] -= 2*np.pi

plt.figure(figsize=(15, 10))
ax = plt.subplot(111, projection='mollweide')
ax.scatter(rosat_ra_rad, np.radians(rosat_dec),
          color='gray', s=1, alpha=0.3, label='ROSAT Sources')

sector_files = glob.glob('sector*_interesting.txt')
colors = plt.cm.rainbow(np.linspace(0, 1, len(sector_files)))

for file, color in zip(sector_files, colors):
    data = pd.read_csv(file, delim_whitespace=True)
    ra_degrees = data['RA'] * (180.0/12.0)
    coords = SkyCoord(ra=ra_degrees*u.degree, dec=data['DEC']*u.degree)
    ra_rad = coords.ra.wrap_at(180*u.deg).radian
    dec_rad = coords.dec.radian
    sector_num = file.split('_')[0]
    ax.scatter(ra_rad, dec_rad, s=10, alpha=0.9, c=[color], 
              label=sector_num, marker='o')
    ax.grid(True) 
plt.title('ROSAT Sources and TESS Sectors Sky Distribution')
ax.set_xlabel('Right Ascension')
ax.set_ylabel('Declination')
plt.legend(bbox_to_anchor=(1.12, 1), loc='upper right')
plt.savefig('ROSAT_TESS_sectors_map.png', dpi=300, bbox_inches='tight')
plt.close()
