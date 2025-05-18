from astropy.io import fits

catalog_files = [
    'eFEDS_c001_hard_V6.2.fits.gz',
    'eFEDS_c001_main_V6.2.fits.gz',
    'etaCha_c001_hard_V1.fits.gz',
    'etaCha_c001_main_V1.fits.gz'
]

for catalog_path in catalog_files:
    print(f"\nColumns in {catalog_path}:")
    with fits.open(catalog_path) as hdul:
        columns = hdul[1].columns
        for col in columns:
            print(f"- {col.name}: {col.format}")
