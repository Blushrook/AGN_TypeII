import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import logging

class LightCurveData:
    def __init__(self, directory, save_directory):
        self.directory = directory
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_data(self, filename):
        """
        Load light curve data from a CSV file.
        """
        file_path = os.path.join(self.directory, filename)
        try:
            data = pd.read_csv(file_path, sep=r'\s+')
            data.columns = data.columns.str.strip()
            data = data.replace('-', pd.NA).astype(float)
            return data
        except FileNotFoundError as e:
            logging.error(f"Failed to find the file {filename}: {e}")
        except Exception as e:
            logging.error(f"An error occurred while reading {filename}: {e}")
        return None

    def plot_light_curve(self, data, title='Light Curve', filename='plot.png'):
        """
        Plot light curve data.
        """
        save_path = os.path.join(self.save_directory, filename)
        if os.path.exists(save_path):
            logging.info(f"Plot already exists: {save_path}")
            return

        if data is not None and not data.empty:
            try:
                plt.figure(figsize=(10, 6))
                plt.errorbar(data['BTJD'], data['cts'], yerr=data['e_cts'], fmt='o', color='blue', ecolor='lightgray', elinewidth=3, capsize=0)
                plt.title(title)
                plt.xlabel('BTJD (Barycentric TESS Julian Date)')
                plt.ylabel('Counts (cts)')
                plt.grid(True)
                plt.savefig(save_path)
                plt.close()
                logging.info(f"Plot saved to {save_path}")
            except KeyError as e:
                logging.error(f"Key error: {e} - Check that 'BTJD', 'cts', and 'e_cts' are in your DataFrame")
            except Exception as e:
                logging.error(f"An error occurred while plotting {title}: {e}")
        else:
            logging.warning("No data to plot.")

    def calculate_weighted_standard_deviation(self, data):
        """
        Calculate weighted standard deviation for the light curve data.
        """
        if 'cts' in data.columns and 'e_cts' in data.columns:
            flux = data['cts']
            flux_uncertainty = data['e_cts']
            mean_flux = flux.mean()
            stddev = flux.std(ddof=1)
            weights = 1 / (flux_uncertainty**2)
            weighted_mean = np.sum(weights * flux) / np.sum(weights)
            weighted_variance = np.sum(weights * (flux - weighted_mean) **2) / np.sum(weights)
            weighted_stddev = np.sqrt(weighted_variance)

            return mean_flux, stddev, weighted_stddev
        else:
            logging.error("Columns 'cts' or 'e_cts' not found in the data.")
            return None, None, None

