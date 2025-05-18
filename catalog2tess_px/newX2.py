import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from scipy import stats
import pandas as pd
from catalogs.HyperLedaCsv import HyperLedaCsv
from tabulate import tabulate

SECTORS = ['02', '04', '05', '06', '07', '19', '20', '21']

class LightCurveAnalysis:
    def __init__(self, directory, save_directory):
        self.root_directory = directory
        self.save_directory = save_directory
        self.agn_types = ['S1.5', 'S1.6', 'S1.7', 'S1.8', 'S1.9', 'S2']
        self.catalogs = {}
        self.sector_data = {}
        
        for sector in SECTORS:
            self.sector_data[sector] = {
                'chi2_test1': [],
                'chi2_test2': []
            }
            
            for cam in range(1, 5):
                catalog_path = f'/home/kicowlin/SummerResearch2024/catalog2tess_px/HyperLEDA/s{sector}/hyperleda_s{sector}_cam{cam}.txt'
                if os.path.exists(catalog_path):
                    if sector not in self.catalogs:
                        self.catalogs[sector] = {}
                    self.catalogs[sector][cam] = HyperLedaCsv(catalog_path)

    def get_agn_type(self, obj_name, sector, camera):
        if sector in self.catalogs and camera in self.catalogs[sector]:
            catalog = self.catalogs[sector][camera]
            mask = catalog.objname == obj_name
            if any(mask):
                return catalog.agnclass[mask][0]
        return None

    def get_data_directories(self, sector):
        directories = []
        for cam in range(1, 5):
            for ccd in range(1, 5):
                path = os.path.join(self.root_directory, f'sector{sector}/cam{cam}_ccd{ccd}/lc_hyperleda')
                if os.path.exists(path):
                    directories.append((path, cam))
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
        
        # Filter out values over 5000
        if reduced_chi2_1 > 5000:
            reduced_chi2_1 = np.nan
            
        test2_data = clipped_data['cts'] / clipped_data['e_cts']
        chi2_2 = np.sum((test2_data) ** 2)
        dof_2 = N
        reduced_chi2_2 = chi2_2 / dof_2
        
        # Filter out values over 5000
        if reduced_chi2_2 > 5000:
            reduced_chi2_2 = np.nan
            
        return (test1_data, chi2_1, reduced_chi2_1), (test2_data, chi2_2, reduced_chi2_2)

    def plot_all_sectors_chi_square_comparison(self):
        plt.figure(figsize=(12, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(SECTORS)))
        
        # Collect all valid data points
        all_chi2_test1 = []
        all_chi2_test2 = []
        
        for s in SECTORS:
            if s in self.sector_data and len(self.sector_data[s]['chi2_test1']) > 0:
                test1 = np.array(self.sector_data[s]['chi2_test1'])
                test2 = np.array(self.sector_data[s]['chi2_test2'])
                mask = (test1 < 5000) & (test2 < 5000) & ~np.isnan(test1) & ~np.isnan(test2)
                all_chi2_test1.extend(test1[mask])
                all_chi2_test2.extend(test2[mask])
        
        all_chi2_test1 = np.array(all_chi2_test1)
        all_chi2_test2 = np.array(all_chi2_test2)
        
        if len(all_chi2_test1) > 0 and len(all_chi2_test2) > 0:
            percentile_95_test1 = np.percentile(all_chi2_test1, 95)
            percentile_95_test2 = np.percentile(all_chi2_test2, 95)
            
            # Plot each sector with different colors
            for i, sector in enumerate(SECTORS):
                if sector in self.sector_data and len(self.sector_data[sector]['chi2_test1']) > 0:
                    chi2_test1 = np.array(self.sector_data[sector]['chi2_test1'])
                    chi2_test2 = np.array(self.sector_data[sector]['chi2_test2'])
                    
                    # Filter out values over 5000 and NaNs
                    mask = (chi2_test1 < 5000) & (chi2_test2 < 5000) & ~np.isnan(chi2_test1) & ~np.isnan(chi2_test2)
                    chi2_test1 = chi2_test1[mask]
                    chi2_test2 = chi2_test2[mask]
                    
                    if len(chi2_test1) > 0:
                        # Switch axes: x=chi2_test2, y=chi2_test1
                        plt.scatter(chi2_test2, chi2_test1, alpha=0.6, color=colors[i],
                                   label=f'Sector {sector}', s=30)
            
            # Switch axes for the percentile lines
            plt.axhline(y=percentile_95_test1, color='black', linestyle='--',
                    label=f'95th percentile: {percentile_95_test1:.2f}')
            plt.axvline(x=percentile_95_test2, color='black', linestyle=':',
                    label=f'95th percentile: {percentile_95_test2:.2f}')
            
            # Switch axis labels
            plt.ylabel('Reduced χ²: Standard Deviation')
            plt.xlabel('Reduced χ²: Errors')
            plt.title('Chi-Square Comparison : All Sectors')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_directory, 'all_sectors_chi2_comparison.png'))
        plt.close()

    def plot_zoomed_all_sectors_chi_square(self):
        # Collect all valid data points
        all_chi2_test1 = []
        all_chi2_test2 = []
        
        for s in SECTORS:
            if s in self.sector_data and len(self.sector_data[s]['chi2_test1']) > 0:
                test1 = np.array(self.sector_data[s]['chi2_test1'])
                test2 = np.array(self.sector_data[s]['chi2_test2'])
                mask = (test1 < 5000) & (test2 < 5000) & ~np.isnan(test1) & ~np.isnan(test2)
                all_chi2_test1.extend(test1[mask])
                all_chi2_test2.extend(test2[mask])
        
        all_chi2_test1 = np.array(all_chi2_test1)
        all_chi2_test2 = np.array(all_chi2_test2)
        
        if len(all_chi2_test1) > 0 and len(all_chi2_test2) > 0:
            percentile_85_test1 = np.percentile(all_chi2_test1, 85)
            percentile_85_test2 = np.percentile(all_chi2_test2, 85)
            
            plt.figure(figsize=(12, 10))
            colors = plt.cm.tab10(np.linspace(0, 1, len(SECTORS)))
            
            for i, sector in enumerate(SECTORS):
                if sector in self.sector_data and len(self.sector_data[sector]['chi2_test1']) > 0:
                    chi2_test1 = np.array(self.sector_data[sector]['chi2_test1'])
                    chi2_test2 = np.array(self.sector_data[sector]['chi2_test2'])
                    
                    # Filter out values over 5000, NaNs, and apply 85th percentile filter
                    mask = (chi2_test1 < 5000) & (chi2_test2 < 5000) & \
                           (chi2_test1 <= percentile_85_test1) & (chi2_test2 <= percentile_85_test2) & \
                           ~np.isnan(chi2_test1) & ~np.isnan(chi2_test2)
                    
                    if np.any(mask):
                        # Switch axes: x=chi2_test2, y=chi2_test1
                        plt.scatter(chi2_test2[mask], chi2_test1[mask], alpha=0.6, color=colors[i],
                                   label=f'Sector {sector}', s=30)
            
            # Switch axes for the percentile lines
            plt.axhline(y=percentile_85_test1, color='black', linestyle='--',
                       label=f'85th percentile: {percentile_85_test1:.2f}')
            plt.axvline(x=percentile_85_test2, color='black', linestyle=':',
                       label=f'85th percentile: {percentile_85_test2:.2f}')
            
            plt.ylabel('Reduced χ²: Standard Deviation')
            plt.xlabel('Reduced χ²: Errors')
            plt.title('Zoomed Chi-Square Comparison Across All Sectors (≤85th percentile)')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_directory, 'all_sectors_chi2_comparison_zoomed.png'))
        plt.close()

    def plot_chi_square_comparison(self, chi2_test1, chi2_test2):
        # Filter out values over 5000 and NaNs
        mask = (chi2_test1 < 5000) & (chi2_test2 < 5000) & ~np.isnan(chi2_test1) & ~np.isnan(chi2_test2)
        chi2_test1 = chi2_test1[mask]
        chi2_test2 = chi2_test2[mask]
        
        if len(chi2_test1) == 0 or len(chi2_test2) == 0:
            print("No valid data points for chi-square comparison plot")
            return
        
        fig = plt.figure(figsize=(10, 10))
        gs = plt.GridSpec(3, 3)
        
        ax_scatter = fig.add_subplot(gs[1:, :-1])
        # Switch axes: x=chi2_test2, y=chi2_test1
        ax_scatter.scatter(chi2_test2, chi2_test1, alpha=0.5)
        
        percentile_95_test1 = np.percentile(chi2_test1, 95)
        percentile_95_test2 = np.percentile(chi2_test2, 95)
        
        # Switch axes for percentile lines
        ax_scatter.axhline(y=percentile_95_test1, color='r', linestyle='--',
                        label=f'95th percentile: {percentile_95_test1:.2f}')
        ax_scatter.axvline(x=percentile_95_test2, color='g', linestyle='--',
                        label=f'95th percentile: {percentile_95_test2:.2f}')
        
        # Switch axis labels
        ax_scatter.set_ylabel('Reduced χ²: Standard Deviation')
        ax_scatter.set_xlabel('Reduced χ²: Errors')
        ax_scatter.legend()
        ax_scatter.grid(True)
        
        # Top CDF (now for chi2_test1)
        ax_cdf_top = fig.add_subplot(gs[0, :-1])
        counts1, bins1, _ = ax_cdf_top.hist(chi2_test1, bins=30, density=True, alpha=0)
        cumulative1 = np.cumsum(counts1)
        cumulative1 = cumulative1 / cumulative1[-1]
        bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
        ax_cdf_top.plot(bin_centers1, cumulative1, 'b-', label='CDF')
        ax_cdf_top.scatter(bin_centers1, cumulative1, alpha=0.5, s=10, c='red', label='Data')
        
        # Mark the 95th percentile on the CDF
        idx_95 = np.searchsorted(cumulative1, 0.95)
        if idx_95 < len(bin_centers1):
            ax_cdf_top.axvline(x=bin_centers1[idx_95], color='r', linestyle='--')
            ax_cdf_top.text(bin_centers1[idx_95], 0.5, '95%', rotation=90, verticalalignment='center')
        
        ax_cdf_top.set_xticklabels([])
        ax_cdf_top.set_ylabel('CDF (StdDev)')
        ax_cdf_top.set_ylim(0, 1)
        ax_cdf_top.legend()
        ax_cdf_top.grid(True)
        
        # Right CDF (now for chi2_test2)
        ax_cdf_right = fig
        # Right CDF (now for chi2_test2)
        ax_cdf_right = fig.add_subplot(gs[1:, -1])
        counts2, bins2, _ = ax_cdf_right.hist(chi2_test2, bins=30, density=True, alpha=0, orientation='horizontal')
        cumulative2 = np.cumsum(counts2)
        cumulative2 = cumulative2 / cumulative2[-1]
        bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
        ax_cdf_right.plot(cumulative2, bin_centers2, 'b-', label='CDF')
        ax_cdf_right.scatter(cumulative2, bin_centers2, alpha=0.5, s=10, c='red', label='Data')
        
        # Mark the 95th percentile on the CDF
        idx_95 = np.searchsorted(cumulative2, 0.95)
        if idx_95 < len(bin_centers2):
            ax_cdf_right.axhline(y=bin_centers2[idx_95], color='g', linestyle='--')
            ax_cdf_right.text(0.5, bin_centers2[idx_95], '95%', horizontalalignment='center')
        
        ax_cdf_right.set_yticklabels([])
        ax_cdf_right.set_xlabel('CDF (Errors)')
        ax_cdf_right.set_xlim(0, 1)
        ax_cdf_right.legend()
        ax_cdf_right.grid(True)
        
        plt.suptitle('Chi-Square Comparison with CDFs')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_directory, 'chi2_comparison.png'))
        plt.close()

    def plot_zoomed_chi_square(self, chi2_test1, chi2_test2):
        # Filter out values over 5000 and NaNs
        mask = (chi2_test1 < 5000) & (chi2_test2 < 5000) & ~np.isnan(chi2_test1) & ~np.isnan(chi2_test2)
        chi2_test1 = chi2_test1[mask]
        chi2_test2 = chi2_test2[mask]
        
        if len(chi2_test1) == 0 or len(chi2_test2) == 0:
            print("No valid data points for zoomed chi-square plot")
            return
        
        percentile_85_test1 = np.percentile(chi2_test1, 85)
        percentile_85_test2 = np.percentile(chi2_test2, 85)
        
        # Apply 85th percentile filter
        mask = (chi2_test1 <= percentile_85_test1) & (chi2_test2 <= percentile_85_test2)
        zoomed_chi2_test1 = chi2_test1[mask]
        zoomed_chi2_test2 = chi2_test2[mask]
        
        plt.figure(figsize=(10, 10))
        # Switch axes: x=chi2_test2, y=chi2_test1
        plt.scatter(zoomed_chi2_test2, zoomed_chi2_test1, alpha=0.5)
        
        # Switch axes for percentile lines
        plt.axhline(y=percentile_85_test1, color='r', linestyle='--',
                    label=f'85th percentile: {percentile_85_test1:.2f}')
        plt.axvline(x=percentile_85_test2, color='g', linestyle='--',
                    label=f'85th percentile: {percentile_85_test2:.2f}')
        
        plt.ylabel('Reduced χ²: Standard Deviation')
        plt.xlabel('Reduced χ²: Errors')
        plt.title('Zoomed Chi-Square Comparison (≤85th percentile)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_directory, 'chi2_comparison_zoomed.png'))
        plt.close()

def main():
    root_dir = '/home/kicowlin/SummerResearch2024'
    save_dir = '/home/kicowlin/SummerResearch2024/chi_square_results_all_sectors'
    os.makedirs(save_dir, exist_ok=True)
    
    analyzer = LightCurveAnalysis(root_dir, save_dir)
    print("Starting analysis...")
    
    all_reduced_chi2_test1 = []
    all_reduced_chi2_test2 = []
    results_data = []
    
    output_file = os.path.join(save_dir, 'Chi_square_results_all_sectors.txt')
    
    with open(output_file, 'w') as f:
        headers = ["Sector", "Object_Name", "AGN_Type", "Chi2_Flux/StdDev", "Reduced_Chi2_Flux/StdDev",
                  "Chi2_Flux/Errors", "Reduced_Chi2_Flux/Errors"]
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---' for _ in headers]) + ' |\n')
        
        for sector in SECTORS:
            print(f"Processing Sector {sector}...")
            
            sector_output_file = os.path.join(save_dir, f'Chi_square_results_sector{sector}.txt')
            with open(sector_output_file, 'w') as sector_f:
                sector_f.write('| ' + ' | '.join(headers[1:]) + ' |\n')
                sector_f.write('| ' + ' | '.join(['---' for _ in headers[1:]]) + ' |\n')
            
            directories = analyzer.get_data_directories(sector)
            print(f"Found {len(directories)} directories to process for Sector {sector}")
            
            for data_dir, camera in directories:
                print(f"Processing directory: {data_dir}")
                files = os.listdir(data_dir)
                print(f"Found {len(files)} files in directory")
                
                for filename in files:
                    if filename.startswith('lc_'):
                        obj_name = filename.replace('lc_', '').replace('_cleaned', '')
                        agn_type = analyzer.get_agn_type(obj_name, sector, camera)
                        
                        if agn_type in analyzer.agn_types:
                            full_path = os.path.join(data_dir, filename)
                            data = analyzer.load_data(full_path)
                            clipped_data = analyzer.sigma_clip_data(data)
                            test1_results, test2_results = analyzer.calculate_chi2(clipped_data)
                            
                            # Skip values over 5000 or NaN
                            if (np.isnan(test1_results[2]) or np.isnan(test2_results[2]) or 
                                test1_results[2] >= 5000 or test2_results[2] >= 5000):
                                continue
                                
                            all_reduced_chi2_test1.append(test1_results[2])
                            all_reduced_chi2_test2.append(test2_results[2])
                            
                            analyzer.sector_data[sector]['chi2_test1'].append(test1_results[2])
                            analyzer.sector_data[sector]['chi2_test2'].append(test2_results[2])
                            
                            row_data = [
                                sector,
                                obj_name,
                                agn_type,
                                f"{test1_results[1]:.4f}",
                                f"{test1_results[2]:.4f}",
                                f"{test2_results[1]:.4f}",
                                f"{test2_results[2]:.4f}"
                            ]
                            results_data.append(row_data)
                            f.write('| ' + ' | '.join(str(item) for item in row_data) + ' |\n')
                            
                            with open(sector_output_file, 'a') as sector_f:
                                sector_f.write('| ' + ' | '.join(str(item) for item in row_data[1:]) + ' |\n')
                            
                            print(f"Results for {filename} (Sector {sector}):")
                            print(f"StdDev - Reduced Chi2: {test1_results[2]:.2f}")
                            print(f"Errors - Reduced Chi2: {test2_results[2]:.2f}")
    
    chi2_test1_array = np.array(all_reduced_chi2_test1)
    chi2_test2_array = np.array(all_reduced_chi2_test2)
    
    # Only generate plots if we have data
    if len(chi2_test1_array) > 0 and len(chi2_test2_array) > 0:
        analyzer.plot_chi_square_comparison(chi2_test1_array, chi2_test2_array)
        analyzer.plot_zoomed_chi_square(chi2_test1_array, chi2_test2_array)
        analyzer.plot_all_sectors_chi_square_comparison()
        analyzer.plot_zoomed_all_sectors_chi_square()
    else:
        print("No valid data points found for plotting")
    
    print(f"Analysis complete. Results saved to {output_file}")
    print(f"Plots saved in directory: {save_dir}")

if __name__ == "__main__":
    main()

