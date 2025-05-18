import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

filename = 'eRASS1_Hard.v1.0.fits'

with fits.open(filename) as hdul:
    data = hdul[1].data
    
ra = data['RA']
dec = data['DEC']

print(f"RA range: {np.min(ra):.5f} to {np.max(ra):.5f}")
print(f"DEC range: {np.min(dec):.6f} to {np.max(dec):.6f}")
print(f"Number of sources: {len(ra)}")

ra_rad = np.deg2rad(ra - 180)
dec_rad = np.deg2rad(dec)

plt.figure(figsize=(12, 8))
ax = plt.subplot(111, projection='mollweide')
ax.scatter(ra_rad, dec_rad, s=2, alpha=0.5)
ax.grid(True)
ax.set_xticklabels(['210','240','270','300','330','0','30','60','90','120','150'])
plt.title('eRASSI Sources - Mollweide Projection')

plt.savefig('eRASSI_mollweide.png', dpi=300, bbox_inches='tight')
plt.close()
