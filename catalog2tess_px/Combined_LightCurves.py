import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
from collections import defaultdict
from catalogs.HyperLedaCsv import HyperLedaCsv

class MultiSectorLightCurveData:
    def __init__(self, root_directory, save_directory):
        self.root_directory = root_directory
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Sectors to process
        self.sectors = ['02', '03', '04', '05', '07', '19', '20', '21']
        
        # AGN classes to include (type 1.5 to type 2)
        self.agn_classes = ['S1.5', 'S1.6', 'S1.7', 'S1.8', 'S1.9', 'S2']

    def load_data_from_sector(self, obj_name, sector, cam, ccd):
        """Load light curve data for a specific object from a specific sector"""
        directory = f'{self.root_directory}/sector{sector}/cam{cam}_ccd{ccd}/lc_hyperleda'
        file_path = os.path.join(directory, f"lc_{obj_name}_cleaned")
        
        try:
            if os.path.exists(file_path):
                data = pd.read_csv(file_path, sep=r'\s+')
                data.columns = data.columns.str.strip()
                data = data.replace('-', pd.NA).astype(float)
                # Add sector information to the data
                data['sector'] = int(sector)
                return data
            else:
                logging.debug(f"File not found: {file_path}")
                return None
        except Exception as e:
            logging.error(f"An error occurred while reading {file_path}: {e}")
            return None

    def sigma_clip_data(self, data, sigma=3, maxiters=5):
        """Apply sigma clipping to the data"""
        try:
            if data is None or data.empty:
                return None
                
            data_values = data['cts'].values
            mask = np.ones(len(data_values), dtype=bool)
            for _ in range(maxiters):
                mean = np.mean(data_values[mask])
                std = np.std(data_values[mask])
                new_mask = np.abs(data_values - mean) < sigma * std
                if np.all(new_mask == mask):
                    break
                mask = new_mask
            clipped_data = data.iloc[mask]
            return clipped_data
        except Exception as e:
            logging.error(f"An error occurred during sigma clipping: {e}")
            return None

    def combine_sector_data(self, obj_name, agn_class):
        """Combine data for an object across all sectors"""
        combined_data = []
        
        # Search for the object in all sectors, cameras, and CCDs
        for sector in self.sectors:
            for cam in range(1, 5):
                for ccd in range(1, 5):
                    data = self.load_data_from_sector(obj_name, sector, cam, ccd)
                    if data is not None and not data.empty:
                        clipped_data = self.sigma_clip_data(data)
                        if clipped_data is not None and not clipped_data.empty:
                            combined_data.append(clipped_data)
                            logging.info(f"Found data for {obj_name} in sector {sector}, camera {cam}, CCD {ccd}")
        
        if not combined_data:
            logging.warning(f"No data found for {obj_name} in any sector")
            return None
            
        # Combine all data frames
        combined_df = pd.concat(combined_data, ignore_index=True)
        # Sort by time
        combined_df = combined_df.sort_values('BTJD')
        
        return combined_df

    def plot_combined_light_curve(self, data, obj_name, agn_class):
        """Plot combined light curve across all sectors"""
        if data is None or data.empty:
            logging.warning(f"No data to plot for {obj_name}")
            return
            
        # Create directory for the AGN class if it doesn't exist
        class_save_directory = os.path.join(self.save_directory, agn_class)
        os.makedirs(class_save_directory, exist_ok=True)
        
        # Define save path
        save_path = os.path.join(class_save_directory, f"{obj_name}_combined_light_curve.png")
        
        # Check if the plot already exists
        if os.path.exists(save_path):
            logging.info(f"Combined light curve already exists: {save_path}")
            return
            
        try:
            plt.figure(figsize=(12, 6))
            
            # Group data by sector
            sectors = data['sector'].unique()
            sectors.sort()
            
            # Plot each sector with a different color but connect all points
            plt.errorbar(data['BTJD'], data['cts'], yerr=data['e_cts'], 
                         fmt='o', color='black', ecolor='lightgray', 
                         elinewidth=1, capsize=2, markersize=4)
            
            # Add sector labels
            for sector in sectors:
                sector_data = data[data['sector'] == sector]
                mid_point = len(sector_data) // 2
                if not sector_data.empty and mid_point < len(sector_data):
                    x_pos = sector_data.iloc[mid_point]['BTJD']
                    y_max = sector_data['cts'].max()
                    plt.text(x_pos, y_max * 1.05, f"S{sector}", 
                             horizontalalignment='center', fontsize=8)
            
            plt.title(f"{obj_name} - {agn_class} - Combined Light Curve")
            plt.xlabel('BTJD (Barycentric TESS Julian Date)')
            plt.ylabel('Counts (cts)')
            plt.grid(True, alpha=0.3)
            
            # Add a note about the sectors included
            sector_str = ", ".join([f"S{s}" for s in sectors])
            plt.figtext(0.5, 0.01, f"Sectors: {sector_str}", 
                        ha="center", fontsize=8, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            logging.info(f"Combined light curve saved to {save_path}")
            
        except Exception as e:
            logging.error(f"An error occurred while plotting {obj_name}: {e}")

def process_agn_objects(root_directory, save_directory):
    """Process all AGN objects and create combined light curves"""
    plotter = MultiSectorLightCurveData(root_directory, save_directory)
    
    # Dictionary to store objects by AGN class
    objects_by_class = defaultdict(list)
    
    # Find all objects in all sectors
    for sector in plotter.sectors:
        for cam in range(1, 5):
            # Load catalog for this camera using HyperLedaCsv
            catalog_file = f"{root_directory}/catalog2tess_px/HyperLEDA/s{sector}/hyperleda_s{sector}_cam{cam}.txt"
            if not os.path.exists(catalog_file):
                logging.warning(f"Catalog file not found: {catalog_file}")
                continue
                
            try:
                # Use the HyperLedaCsv class to read the catalog
                cat = HyperLedaCsv(catalog_file)
                
                # Filter for AGN classes we're interested in
                for agn_class in plotter.agn_classes:
                    mask = cat.agnclass == agn_class
                    if any(mask):
                        for obj_name in cat.objname[mask]:
                            if obj_name not in objects_by_class[agn_class]:
                                objects_by_class[agn_class].append(obj_name)
                                logging.info(f"Found {agn_class} object: {obj_name}")
            except Exception as e:
                logging.error(f"Error processing catalog {catalog_file}: {e}")
    
    # Process each object
    for agn_class, obj_names in objects_by_class.items():
        logging.info(f"Processing {len(obj_names)} objects for class {agn_class}")
        for obj_name in obj_names:
            logging.info(f"Processing {obj_name} ({agn_class})")
            combined_data = plotter.combine_sector_data(obj_name, agn_class)
            if combined_data is not None and len(combined_data) > 0:
                plotter.plot_combined_light_curve(combined_data, obj_name, agn_class)

if __name__ == "__main__":
    root_directory = '/home/kicowlin/SummerResearch2024'
    save_directory = f'{root_directory}/combined_plots/CombinedLightCurves'
    process_agn_objects(root_directory, save_directory)

