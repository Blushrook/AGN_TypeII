import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from scipy import stats
from LightCurve2 import LightCurveData

class HistoGaussData:
    def __init__(self, directory, save_directory, light_curve_data):
        self.directory = directory
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.light_curve_data = light_curve_data

    def plot_histogram(self, clipped_data, title, filename, num_bins=30):
        clipped_data = clipped_data[np.isfinite(clipped_data)]
        if len(clipped_data) == 0:
            logging.warning(f"No finite data points for {title}")
            return None, None, None, None
        
        plt.figure(figsize=(10, 6))
        n, bins, _ = plt.hist(clipped_data, bins=num_bins, density=True, alpha=0.7)

        mu, sigma = stats.norm.fit(clipped_data)
        x = np.linspace(np.min(clipped_data), np.max(clipped_data), 100)
        gaussian = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, gaussian, 'r-', lw=2, label='Gaussian fit')

        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(self.save_directory, filename))
        plt.close()

        return clipped_data, n, bins, mu, sigma

    def calculate_chi2(self, observed, expected, errors):
        chi2 = np.sum(((observed-expected) / errors) **2)
        dof = len(observed) - 1
        reduced_chi2 = chi2 / dof
        return chi2, reduced_chi2

    def calculate_and_plot_histograms(self, clipped_data, obj_name, filename, num_bins=100):
        mean_flux, stddev, weighted_stddev = self.light_curve_data.calculate_weighted_standard_deviation(clipped_data)
        test1_data = clipped_data['cts'] / stddev
        non_finite = np.sum(~np.isfinite(test1_data))
        if non_finite > 0:
            logging.warning(f"{obj_name}: Test 1 data contains {non_finite} non-finite values")
            test1_data = test1_data[~np.isfinite(test1_data)]
        _, n1, bins1, mu1, sigma1 = self.plot_histogram(
            test1_data,
            f"Test 1 Histogram for {obj_name}",
            f"{obj_name}_test1_histogram.png",
            num_bins
        )
        observed_1 = test1_data
        expected_1 = np.full_like(observed_1, mu1)
        errors_1 = np.full_like(observed_1, sigma1)
        chi2_1, reduced_chi2_1 = self.calculate_chi2(observed_1, expected_1, errors_1)


        test2_data = clipped_data['cts'] / clipped_data['e_cts']
        non_finite = np.sum(~np.isfinite(test2_data))
        if non_finite > 0:
            logging.warning(f"{obj_name}: Test 2 data contains {non_finite} non-finite values")
            test2_data = test2_data[np.isfinite(test2_data)]

        _, n2, bins2, mu2, sigma2 = self.plot_histogram(
            test2_data,
            f"Test 2 Histogram for {obj_name}",
            f"{obj_name}_test2_histogram.png",
            num_bins
        )
        
        observed_2 = test2_data
        expected_2 = np.zeros_like(observed_2)
        errors_2 = np.ones_like(observed_2)
        chi2_2, reduced_chi2_2 = self.calculate_chi2(observed_2, expected_2, errors_2)
    
        
        return (chi2_1, reduced_chi2_1), (chi2_2, reduced_chi2_2), mean_flux, stddev
        return None, None, None, None
