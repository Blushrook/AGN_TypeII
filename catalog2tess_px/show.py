from astropy.io import fits


catalogs = {
    'eFEDS_hard': 'eFEDS_c001_hard_V6.2.fits.gz',
    'eFEDS_main': 'eFEDS_c001_main_V6.2.fits.gz',
    'etaCha_hard': 'etaCha_c001_hard_V1.fits.gz',
    'etaCha_main': 'etaCha_c001_main_V1.fits.gz'
}


for name, filename in catalogs.items():
    with fits.open(filename) as hdul:
        data = hdul[1].data
        print(f"\nCatalog: {name}")
        print(f"RA range: {data['RA'].min():.3f} to {data['RA'].max():.3f}")
        print(f"DEC range: {data['DEC'].min():.3f} to {data['DEC'].max():.3f}")
        print(f"Number of sources: {len(data['RA'])}")
