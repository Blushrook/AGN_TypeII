from astroquery.vizier import Vizier
import numpy as np
import pandas as pd
import glob

Vizier.ROW_LIMIT = -1
Vizier.columns = ['all']

results = []
sector_files = glob.glob('sector*_interesting.txt')

for file in sector_files:
    data = pd.read_csv(file, delim_whitespace=True)

    for idx, row in data.iterrows():
        ra_deg = row['RA'] * (180.0/12.0)
        result = Vizier.query_region(f"{ra_deg} {row['DEC']}", 
                                   radius='10s', 
                                   catalog='J/A+A/588/A103')
        
        obj_data = {
            'Name': row['Name'],
            'RA': ra_deg,
            'DEC': row['DEC'],
            'Flux_powerlaw': None,
            'Flux_mekal': None,
            'Flux_blackbody': None,
            'CountRate_powerlaw': None,
            'CountRate_mekal': None,
            'CountRate_blackbody': None
        }
        
        if result and len(result) > 0:
            table = result[0][0]
            obj_data.update({
                'Flux_powerlaw': table['Fluxp'] if 'Fluxp' in table.columns else None,
                'Flux_mekal': table['Fluxm'] if 'Fluxm' in table.columns else None,
                'Flux_blackbody': table['Fluxb'] if 'Fluxb' in table.columns else None,
                'CountRate_powerlaw': table['CTRspep'] if 'CTRspep' in table.columns else None,
                'CountRate_mekal': table['CTRspm'] if 'CTRspm' in table.columns else None,
                'CountRate_blackbody': table['CTRspb'] if 'CTRspb' in table.columns else None
            })
        
        results.append(obj_data)
df = pd.DataFrame(results)
with open('rosat_fluxes.txt', 'w') as f:
    header = ['Name', 'RA', 'DEC', 'Flux_powerlaw', 'Flux_mekal', 'Flux_blackbody', 
              'CountRate_powerlaw', 'CountRate_mekal', 'CountRate_blackbody']
    f.write('\t'.join(header) + '\n')

    for _, row in df.iterrows():
        line = [str(row[col]) if pd.notnull(row[col]) else 'NULL' for col in header]
        f.write('\t'.join(line) + '\n')

total_objects = len(results)
found_objects = len(df.dropna(subset=['Flux_powerlaw', 'Flux_mekal', 'Flux_blackbody'], how='all'))
print(f"Found ROSAT data for {found_objects} out of {total_objects} objects")
print("Results saved to rosat_fluxes.txt")
