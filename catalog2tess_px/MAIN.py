import os
import numpy as np
import logging
from catalogs.HyperLedaCsv import HyperLedaCsv
from LightCurve2 import LightCurveData
from Histogram import HistoGaussData

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

root_directory = '/home/kicowlin/SummerResearch2024'
agn_classes = ['S2', 'S1.5', 'S1.6', 'S1.7', 'S1.8', 'S1.9']

def process_light_curves(cam, ccd, cat, agn_class, directory, save_directory):
    results = []
    mask = cat.agnclass == agn_class

    if not any(mask):
        logging.info(f"No {agn_class} objects found in Camera {cam} CCD {ccd}.")
        return results

    logging.info(f"Processing Camera {cam} CCD {ccd} with {sum(mask)} {agn_class} objects...")

    class_save_directory = os.path.join(save_directory, agn_class)
    os.makedirs(class_save_directory, exist_ok=True)

    light_curve_manager = LightCurveData(directory, save_directory)
    histogram_processor = HistoGaussData(directory, class_save_directory, light_curve_manager)

    for obj_name in cat.objname[mask]:
        logging.info(f"Processing Object: {obj_name}")

        lc_file = f"lc_{obj_name}_cleaned"
        lc_file_path = os.path.join(light_curve_manager.directory, lc_file)

        if not os.path.exists(lc_file_path):
            logging.error(f"Light curve file does not exist: {lc_file_path}")
            continue

        data = light_curve_manager.load_data(lc_file)
        if data is not None:
            clipped_data = light_curve_manager.sigma_clip_data(data)
            if clipped_data is not None:

                chi2_results = histogram_processor.calculate_and_plot_histograms(clipped_data, obj_name, lc_file)

                if chi2_results[0] is None or chi2_results[1] is None:
                    logging.warning(f"Chi-squared values could not be calculated for {obj_name}")
                    continue

                (chi2_normalized, reduced_chi2_normalized), (chi2_standardized, reduced_chi2_standardized), mean_flux, std_dev = chi2_results

                obj_info = {
                    'Name': obj_name,
                    'Objtype': cat.objtype[cat.objname == obj_name][0],
                    'Agnclass': cat.agnclass[cat.objname == obj_name][0],
                    'RA': cat.ra[cat.objname == obj_name][0],
                    'DEC': cat.dec[cat.objname == obj_name][0],
                    'Mean_Flux': mean_flux,
                    'Stddev': std_dev,
                    'Sector': '06',
                    'Camera': f'{cam}',
                    'CCD': f'{ccd}',
                    'Chi2_Normalized': chi2_normalized,
                    'Chi2_reduced_Normalized': reduced_chi2_normalized,
                    'Chi2_Standardized': chi2_standardized,
                    'Chi2_reduced_Standardized': reduced_chi2_standardized
                }
                results.append(obj_info)
                light_curve_manager.plot_light_curve(clipped_data, obj_name + f'LightCurve', f"{obj_name}_LightCurve.png", agn_class)
    return results


def main():
    all_results = []
    for cam in range(1, 5):
        for ccd in range(1, 5):
            camera_files = [f"HyperLEDA/s06/hyperleda_s06_cam{cam}.txt"]
            directory = f'{root_directory}/sector06/cam{cam}_ccd{ccd}/lc_hyperleda'
            save_directory = f'{root_directory}/plots/Sector06'

            light_curve_manager = LightCurveData(directory, save_directory)

            for cam_file in camera_files:
                cat = HyperLedaCsv(cam_file)
                for agn_class in agn_classes:
                    results = process_light_curves(cam, ccd, cat, agn_class, directory, save_directory)
                    all_results.extend(results)
                    logging.debug(f"Current all_results length: {len(all_results)}")

    
    if all_results:
        headers = ["Name", "Objtype", "Agnclass", "RA", "DEC", "Mean_Flux", "Stddev", 
                   "Sector", "Camera", "CCD", "Chi2_Normalized", "Chi2_reduced_Normalized", 
                   "Chi2_Standardized", "Chi2_reduced_Standardized"]
    
        max_widths = {header: len(header) for header in headers}
        for result in all_results:
            for header in headers:
                value = str(result.get(header, ''))
                max_widths[header] = max(max_widths[header], len(value))
        
        with open('processed_light_curves_sector06.txt', 'w') as file:
            header_line = "".join(f"{header:<{max_widths[header] + 2}}" for header in headers)
            file.write(header_line + "\n")
            
            for result in all_results:
                row = "".join(f"{str(result.get(header, '')):<{max_widths[header] + 2}}" for header in headers)
                file.write(row + "\n")

if __name__ == "__main__":
    main()
