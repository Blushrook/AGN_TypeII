import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from scipy import stats
import pandas as pd
from catalogs.HyperLedaCsv import HyperLedaCsv
from tabulate import tabulate 

sector = '07'

class LightCurveAnalysis:
    def __init__(self, directory, save_directory):
        self.root_directory = directory
        self.save_directory = save_directory
        self.agn_types = ['S1.5', 'S1.6', 'S1.7', 'S1.8', 'S1.9', 'S2']
        self.catalogs = {}
        for cam in range(1, 5):
            catalog_path = f'/home/kicowlin/SummerResearch2024/catalog2tess_px/HyperLEDA/s{sector}/hyperleda_s{sector}_cam{cam}.txt'
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
                path = os.path.join(self.root_directory, f'Sector{sector}/cam{cam}_ccd{ccd}/lc_hyperleda')
                if os.path.exists(path):
                    directories.append(path)
        return directories

    def load_data(self, filename):
        data = pd.read_csv(filename, sep=r'\s+')
        data.columns = data.columns.str.strip()
        return data

    def sigma_clip_data(self, data, sigma=3, maxiters=5):
        data_values = data['cts'].values
        mask = np.ones(len(data_values), dtype=bool)
    
        for _ in range(maxiters):
            mean = np.mean(data_values[mask])
            std = np.std(data_values[mask])
            new_mask = np.abs(data_values - mean) < sigma * std
        
            if np.all(new_mask == mask):
                break
            
            mask = new_mask
    
        return data.iloc[mask]

    def calculate_chi2(self, clipped_data):

        N = len(clipped_data['cts']) 
        mean_flux = np.mean(clipped_data['cts'])
        stddev = np.std(clipped_data['cts'], ddof=1)
        
        test1_data = clipped_data['cts'] / stddev
        chi2_1 = np.sum(test1_data ** 2)
        dof_1 = N - 1
        reduced_chi2_1 = chi2_1 / dof_1

        test2_data = clipped_data['cts'] / clipped_data['e_cts']
        chi2_2 = np.sum((test2_data) ** 2)
        dof_2 = N
        reduced_chi2_2 = chi2_2 / dof_2

        return (test1_data, chi2_1, reduced_chi2_1), (test2_data, chi2_2, reduced_chi2_2)


    def plot_cdf(self, data, title, save_path, filename):
        data = np.sort(data)    
        cdf = np.arange(1, len(data) + 1) / len(data)   

        plt.figure(figsize=(10, 6)) 
        plt.plot(data, cdf, marker='.', linestyle='none')   
        plt.title(title)    
        plt.xlabel('Data')
        plt.ylabel('CDF')   
        plt.ylim(0,1)
        plt.yscale('log')   
        plt.savefig(os.path.join(save_path, filename))
        plt.close()
  
    def plot_combined_histograms(self, all_reduced_chi2_test1, all_reduced_chi2_test2):
        plt.figure(figsize=(12, 6))

        plt.subplot(121)
        mu1, sigma1 = stats.norm.fit(all_reduced_chi2_test1)
        x1 = np.linspace(min(all_reduced_chi2_test1), max(all_reduced_chi2_test1), 100)
        pdf1 = stats.norm.pdf(x1, mu1, sigma1)
        plt.hist(all_reduced_chi2_test1, bins=30, density=True, alpha=0.7, label='Data')
        plt.plot(x1, pdf1, 'r-', label=f'PDF\nμ={mu1:.2f}, σ={sigma1:.2f}')
        plt.title('Combined Test 1 Reduced Chi-Square')
        plt.xlabel('Reduced Chi-Square')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)

        plt.subplot(122)
        mu2, sigma2 = stats.norm.fit(all_reduced_chi2_test2)
        x2 = np.linspace(min(all_reduced_chi2_test2), max(all_reduced_chi2_test2), 100)
        pdf2 = stats.norm.pdf(x2, mu2, sigma2)
        plt.hist(all_reduced_chi2_test2, bins=30, density=True, alpha=0.7, label='Data')
        plt.plot(x2, pdf2, 'r-', label=f'PDF\nμ={mu2:.2f}, σ={sigma2:.2f}')
        plt.title('Combined Test 2 Reduced Chi-Square')
        plt.xlabel('Reduced Chi-Square')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_directory, 'combined_reduced_chi2_pdfs.png'))
        plt.close()

    def plot_chi_square_comparison(self, chi2_test1, chi2_test2):
        fig = plt.figure(figsize=(10, 10))
        gs = plt.GridSpec(3, 3)
    
        ax_scatter = fig.add_subplot(gs[1:, :-1])
        ax_scatter.scatter(chi2_test1, chi2_test2, alpha=0.5)
    
        percentile_95_test1 = np.percentile(chi2_test1, 95)
        percentile_95_test2 = np.percentile(chi2_test2, 95)
    
        ax_scatter.axvline(x=percentile_95_test1, color='r', linestyle='--',
                        label=f'95th percentile Test1: {percentile_95_test1:.2f}')
        ax_scatter.axhline(y=percentile_95_test2, color='g', linestyle='--',
                        label=f'95th percentile Test2: {percentile_95_test2:.2f}')
    
        ax_scatter.set_xlabel('Chi-Square Test1')
        ax_scatter.set_ylabel('Chi-Square Test2')
        ax_scatter.legend()
        ax_scatter.grid(True)
    
        ax_cdf_top = fig.add_subplot(gs[0, :-1])
        counts1, bins1, _ = ax_cdf_top.hist(chi2_test1, bins=30, density=True, alpha=0)
        cumulative1 = np.cumsum(counts1)
        cumulative1 = cumulative1 / cumulative1[-1]
        bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
        ax_cdf_top.plot(bin_centers1, cumulative1, 'b-', label='CDF')
        ax_cdf_top.scatter(bin_centers1, cumulative1, alpha=0.5, s=10, c='red', label='Data')
        ax_cdf_top.set_xticklabels([])
        ax_cdf_top.set_ylabel('CDF Test1')
        ax_cdf_top.set_ylim(0, 1) 
        ax_cdf_top.legend()
        ax_cdf_top.grid(True)
    
        ax_cdf_right = fig.add_subplot(gs[1:, -1])
        counts2, bins2, _ = ax_cdf_right.hist(chi2_test2, bins=30, density=True, alpha=0, orientation='horizontal')
        cumulative2 = np.cumsum(counts2)
        cumulative2 = cumulative2 / cumulative2[-1]
        bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
        ax_cdf_right.plot(cumulative2, bin_centers2, 'b-', label='CDF')
        ax_cdf_right.scatter(cumulative2, bin_centers2, alpha=0.5, s=10, c='red', label='Data')
        ax_cdf_right.set_yticklabels([])
        ax_cdf_right.set_xlabel('CDF Test2')
        ax_cdf_right.set_xlim(0, 1)
        ax_cdf_right.legend()
        ax_cdf_right.grid(True)
    
        plt.suptitle('Chi-Square Comparison with CDFs')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_directory, 'chi2_comparison.png'))
        plt.close()

    def plot_zoomed_chi_square(self, chi2_test1, chi2_test2):
        percentile_95_test1 = np.percentile(chi2_test1, 85)
        percentile_95_test2 = np.percentile(chi2_test2, 85)
        mask = (chi2_test1 <= percentile_95_test1) & (chi2_test2 <= percentile_95_test2)
        zoomed_chi2_test1 = chi2_test1[mask]
        zoomed_chi2_test2 = chi2_test2[mask]
    
        plt.figure(figsize=(10, 10))
        plt.scatter(zoomed_chi2_test1, zoomed_chi2_test2, alpha=0.5)
    
        plt.axvline(x=percentile_95_test1, color='r', linestyle='--',
                    label=f'95th percentile Test1: {percentile_95_test1:.2f}')
        plt.axhline(y=percentile_95_test2, color='g', linestyle='--',
                    label=f'95th percentile Test2: {percentile_95_test2:.2f}')
    
        plt.xlabel('Chi-Square Test1')
        plt.ylabel('Chi-Square Test2')
        plt.title('Zoomed Chi-Square Comparison (≤95th percentile)')
        plt.legend()
        plt.grid(True)
    
        plt.savefig(os.path.join(self.save_directory, 'chi2_comparison_zoomed.png'))
        plt.close()

def main():
    root_dir = '/home/kicowlin/SummerResearch2024'
    save_dir = '/home/kicowlin/SummerResearch2024/chi_square_results_s07'
    output_file = os.path.join(save_dir, 'Chi_square_results_sector08.txt')
    os. makedirs(save_dir, exist_ok=True)

    analyzer = LightCurveAnalysis(root_dir, save_dir)
    print("Starting analysis...")
    
    all_reduced_chi2_test1 = []
    all_reduced_chi2_test2 = []
    results_data = []

    with open(output_file, 'w') as f:
        headers = ["Object_Name", "AGN_Type", "Chi2_Test1", "Reduced_Chi2_Test1",
                  "Chi2_Test2", "Reduced_Chi2_Test2"]
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---' for _ in headers]) + ' |\n')
  
  
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

                        row_data = [
                            obj_name,
                            agn_type,
                            f"{test1_results[1]:.4f}",
                            f"{test1_results[2]:.4f}",
                            f"{test2_results[1]:.4f}",
                            f"{test2_results[2]:.4f}"
                        ]   
                        results_data.append(row_data)
                        f.write('| ' + ' | '.join(str(item) for item in row_data) + ' |\n')

                        print(f"Results for {filename}:")
                        print(f"Test 1 - Reduced Chi2: {test1_results[2]:.2f}")
                        print(f"Test 2 - Reduced Chi2: {test2_results[2]:.2f}")
    
        
    chi2_test1_array = np.array(all_reduced_chi2_test1)
    chi2_test2_array = np.array(all_reduced_chi2_test2)

    
    analyzer.plot_combined_histograms(chi2_test1_array, chi2_test2_array)
    analyzer.plot_chi_square_comparison(chi2_test1_array, chi2_test2_array)
    analyzer.plot_zoomed_chi_square(chi2_test1_array, chi2_test2_array)

    print(f"Analysis complete. Results saved to {output_file}")
    print(f"Plots saved in directory: {save_dir}")

if __name__ == "__main__":
    main()

