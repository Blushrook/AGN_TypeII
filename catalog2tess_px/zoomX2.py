import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import stats
from catalogs.HyperLedaCsv import HyperLedaCsv

class LightCurveAnalysis:
    def __init__(self, directory, save_directory):
        self.root_directory = directory
        self.save_directory = save_directory
        self.agn_types = ['S1.5', 'S1.6', 'S1.7', 'S1.8', 'S1.9', 'S2']
        self.catalogs = {}
        for cam in range(1, 5):
            catalog_path = f'/home/kicowlin/SummerResearch2024/catalog2tess_px/HyperLEDA/s07/hyperleda_s07_cam{cam}.txt'
            self.catalogs[cam] = HyperLedaCsv(catalog_path)

    def get_agn_type(self, obj_name, camera):
        catalog = self.catalogs[camera]
        mask = catalog.objname == obj_name
        if any(mask):
            return catalog.agnclass[mask][0]
        return None

    def get_data_directories(self):
        directories = []
        for cam in range(1, 5):
            for ccd in range(1, 5):
                path = os.path.join(self.root_directory, f'Sector07/cam{cam}_ccd{ccd}/lc_hyperleda')
                if os.path.exists(path):
                    directories.append(path)
        return directories

    def load_data(self, filename):
        data = pd.read_csv(filename, sep=r'\s+')
        data.columns = data.columns.str.strip()
        return data

    def sigma_clip_data(self, data, sigma=3):
        data_values = data['cts'].values
        mean = np.mean(data_values)
        std = np.std(data_values)
        mask = np.abs(data_values - mean) < sigma * std
        return data.iloc[mask]

    def calculate_chi2(self, clipped_data):
        stddev = np.std(clipped_data['cts'])
        test1_data = clipped_data['cts'] / stddev
        chi2_1 = np.sum(test1_data**2)
        reduced_chi2_1 = chi2_1 / (len(test1_data) - 1)

        test2_data = clipped_data['cts'] / clipped_data['e_cts']
        chi2_2 = np.sum(test2_data**2)
        reduced_chi2_2 = chi2_2 / (len(test2_data) - 1)

        return (test1_data, chi2_1, reduced_chi2_1), (test2_data, chi2_2, reduced_chi2_2)

    def plot_chi_square_comparison(self, chi2_test1, chi2_test2):
        percentile_95_test1 = np.percentile(chi2_test1, 95)
        percentile_95_test2 = np.percentile(chi2_test2, 95)
        mask_zoomed = (chi2_test1 <= percentile_95_test1) & (chi2_test2 <= percentile_95_test2)
        plt.figure(figsize=(10, 10))
        plt.scatter(chi2_test1[mask_zoomed], chi2_test2[mask_zoomed], alpha=0.5)
        plt.xlabel('Chi-Square Test1')
        plt.ylabel('Chi-Square Test2')
        plt.title('Zoomed Chi-Square Comparison (Excluding Outliers)')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_directory, 'chi2_comparison_zoomed.png'))
        plt.close()

def main():
    root_dir = '/home/kicowlin/SummerResearch2024'
    save_dir = '/home/kicowlin/SummerResearch2024/chi_square_results_s07'
    analyzer = LightCurveAnalysis(root_dir, save_dir)

    print("Starting analysis...")
    all_reduced_chi2_test1 = []
    all_reduced_chi2_test2 = []

    directories = analyzer.get_data_directories()
    print(f"Found {len(directories)} directories to process")

    for data_dir in directories:
        print(f"Processing directory: {data_dir}")
        camera = int(data_dir.split('cam')[1][0])
        files = os.listdir(data_dir)
        print(f"Found {len(files)} files in directory")

        for filename in files:
            if filename.startswith('lc_'):
                obj_name = filename.replace('lc_', '').replace('_cleaned', '')
                agn_type = analyzer.get_agn_type(obj_name, camera)

                if agn_type in analyzer.agn_types:
                    full_path = os.path.join(data_dir, filename)
                    data = analyzer.load_data(full_path)
                    clipped_data = analyzer.sigma_clip_data(data)
                    test1_results, test2_results = analyzer.calculate_chi2(clipped_data)

                    all_reduced_chi2_test1.append(test1_results[2])
                    all_reduced_chi2_test2.append(test2_results[2])

                    print(f"Results for {filename}:")
                    print(f"Test 1 - Reduced Chi2: {test1_results[2]:.2f}")
                    print(f"Test 2 - Reduced Chi2: {test2_results[2]:.2f}")

    analyzer.plot_chi_square_comparison(np.array(all_reduced_chi2_test1),
                                        np.array(all_reduced_chi2_test2))

if __name__ == "__main__":
    main()
