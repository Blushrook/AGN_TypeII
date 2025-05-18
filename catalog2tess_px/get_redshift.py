from catalogs.HyperLedaCsv import HyperLedaCsv
import glob
import numpy as np

def search_catalogs(object_list):
    results = {}
    c = 299792.0
    pattern = f"HyperLEDA/s05/hyperleda_s05_cam*.txt"
    catalog_files = glob.glob(pattern)

    for catalog_file in catalog_files:
        print(f"Searching for object in {pattern}")
        hyperleda = HyperLedaCsv(catalog_file)
        for obj in object_list:
            if obj not in results:
                print(f"Object not found in {pattern}")
                matches = np.where(hyperleda.objname == obj)[0]
                if len(matches) > 0:
                    print(f"Object found in {pattern}")
                    idx = matches[0]
                    results[obj] = {
                            'redshift' : hyperleda.velocity[idx] / c,
                            'e_redshift' : hyperleda.e_velocity[idx] / c
                             }
    return results

def get_redshifts_from_file(input_file, output_file):
    print("Working")
    with open(input_file, 'r') as f:
        object_list = [line.strip() for line in f.readlines()]
    
    results = search_catalogs(object_list) 

    with open(output_file, 'w') as f:
        print("Writing Redshift Results")
        f.write('Object\tRedshift\tRedshift_Error\n')
        for obj in object_list:
            if obj in results:
                line = f"{obj}\t{results[obj]['redshift']}\t{results[obj]['e_redshift']}\n"
                f.write(line)

input_file = 'List.txt'
output_file = 'redshift_results_s05.txt'
get_redshifts_from_file(input_file, output_file)
print("JOB FINISHED")
