import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import logging
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
from catalogs.HyperLedaCsv import HyperLedaCsv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LightCurveGPR:
    def __init__(self, root_directory):
        self.root_directory = root_directory
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

    def combine_sector_data(self, obj_name, agn_class=None):
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

    def fit_gp_model(self, data, n_restarts=10):
        """Fit a Gaussian Process model with a damped random walk kernel"""
        try:
            # Extract time and flux data
            X = data['BTJD'].values.reshape(-1, 1)
            y = data['cts'].values
            y_err = data['e_cts'].values
            
            # Check for NaN values
            valid_indices = ~np.isnan(X.flatten()) & ~np.isnan(y) & ~np.isnan(y_err)
            if not np.any(valid_indices):
                logging.error("No valid data points after filtering NaNs")
                return None, None, None, None
                
            X = X[valid_indices]
            y = y[valid_indices]
            y_err = y_err[valid_indices]
            
            # Normalize the data for better numerical stability
            self.X_mean = X.mean()
            self.X_std = X.std()
            self.y_mean = y.mean()
            self.y_std = y.std()
            X_norm = (X - self.X_mean) / self.X_std
            y_norm = (y - self.y_mean) / self.y_std
            y_err_norm = y_err / self.y_std
            
            # Define the damped random walk kernel (Ornstein-Uhlenbeck process)
            # This is equivalent to a Matern kernel with nu=1/2
            # We also add a white noise kernel to account for measurement errors
            amplitude = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.1, 10.0))
            # The length scale in the DRW corresponds to the characteristic timescale
            # Typical AGN timescales range from days to months
            length_scale = 10.0  # Initial guess in normalized time units
            # Allow for shorter timescales by lowering the bound to 0.1
            drw_kernel = amplitude * Matern(length_scale=length_scale, length_scale_bounds=(0.1, 100.0), nu=0.5)
            
            # Add white noise kernel to account for measurement errors
            noise_level = np.mean(y_err_norm**2)
            noise_min = np.min(y_err_norm**2) if len(y_err_norm) > 0 else 1e-10
            noise_max = np.max(y_err_norm**2)*10 if len(y_err_norm) > 0 else 1.0
            
            noise_kernel = WhiteKernel(noise_level=noise_level,
                                      noise_level_bounds=(noise_min, noise_max))
            
            # Combined kernel
            kernel = drw_kernel + noise_kernel
            
            # Create and fit the GP model
            gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_norm**2,
                                         n_restarts_optimizer=n_restarts, normalize_y=False)
            gp.fit(X_norm, y_norm)
            logging.info(f"Optimized kernel: {gp.kernel_}")
            return gp, X_norm, y_norm, y_err_norm
        except Exception as e:
            logging.error(f"Error in fit_gp_model: {e}")
            return None, None, None, None

    def extract_kernel_parameters(self, gp_model):
        """Extract the DRW parameters from the fitted GP model"""
        try:
            kernel_params = gp_model.kernel_.get_params()
            # Debug the kernel structure
            logging.debug(f"Kernel structure: {gp_model.kernel_}")
            logging.debug(f"Kernel params: {kernel_params}")
            
            # Initialize parameters
            amplitude = None
            length_scale = None
            noise_level = None
            
            # Extract parameters based on kernel structure
            # This handles the typical structure of k1__k1__constant_value for amplitude
            # and k1__k2__length_scale for the timescale
            for param_name, param_value in kernel_params.items():
                if 'constant_value' in param_name and not 'bounds' in param_name:
                    amplitude = param_value
                    logging.info(f"Found amplitude: {param_name} = {param_value} (type: {type(param_value)})")
                elif 'length_scale' in param_name and not 'bounds' in param_name:
                    length_scale = param_value
                    logging.info(f"Found length_scale: {param_name} = {param_value} (type: {type(param_value)})")
                elif 'noise_level' in param_name and not 'bounds' in param_name:
                    noise_level = param_value
                    logging.info(f"Found noise_level: {param_name} = {param_value} (type: {type(param_value)})")
            
            # Convert normalized timescale to days
            if length_scale is not None:
                logging.info(f"Length scale type: {type(length_scale)}")
                logging.info(f"Length scale value: {length_scale}")
                if isinstance(length_scale, (list, tuple, np.ndarray)):
                    tau_drw_days = length_scale[0] * self.X_std
                    logging.info(f"Using first element of sequence: {length_scale[0]}")
                else:
                    tau_drw_days = length_scale * self.X_std
            else:
                tau_drw_days = None
            
            # Convert normalized amplitude to counts
            if amplitude is not None:
                logging.info(f"Amplitude type: {type(amplitude)}")
                logging.info(f"Amplitude value: {amplitude}")
                if isinstance(amplitude, (list, tuple, np.ndarray)):
                    amplitude_counts = amplitude[0] * self.y_std
                    logging.info(f"Using first element of amplitude: {amplitude[0]}")
                else:
                    amplitude_counts = amplitude * self.y_std
            else:
                amplitude_counts = None
                
            return {
                'amplitude': amplitude_counts,
                'tau_drw_days': tau_drw_days,
                'noise_level': noise_level
            }
        except Exception as e:
            logging.error(f"Error in extract_kernel_parameters: {e}")
            return {'amplitude': None, 'tau_drw_days': None, 'noise_level': None}

    def calculate_statistics(self, data, gp_model, X_norm, y_norm):
        """Calculate comprehensive statistics for the light curve and model"""
        try:
            # Original data statistics
            X = X_norm * self.X_std + self.X_mean
            y = y_norm * self.y_std + self.y_mean
            
            # Time span in days
            time_span = X.max() - X.min()
            
            # Basic statistics with NaN handling
            mean_flux = np.nanmean(y)
            std_flux = np.nanstd(y)
            min_flux = np.nanmin(y)
            max_flux = np.nanmax(y)
            amplitude = max_flux - min_flux
            
            # RMS calculation with NaN handling
            valid_y = ~np.isnan(y)
            if np.any(valid_y):
                rms = np.sqrt(np.nanmean(np.square(y[valid_y] - mean_flux)))
            else:
                rms = np.nan
                
            # Error handling for mean_err calculation
            if 'e_cts' in data.columns and not data['e_cts'].isna().all():
                mean_err = np.nanmean(data['e_cts'].values**2)
                snr = mean_flux / np.sqrt(mean_err) if mean_err > 0 else np.nan
            else:
                mean_err = np.nan
                snr = np.nan
            
            # Fractional variability (Fvar) with NaN handling
            variance = np.nanvar(y)
            if 'e_cts' in data.columns and not data['e_cts'].isna().all():
                mean_err_squared = np.nanmean(data['e_cts'].values**2)
            else:
                mean_err_squared = np.nan
                
            excess_variance = variance - mean_err_squared
            if mean_flux != 0 and excess_variance > 0 and not np.isnan(excess_variance):
                frac_var = np.sqrt(excess_variance) / np.abs(mean_flux)
                # Error on fractional variability
                valid_count = np.sum(valid_y)
                if valid_count > 0:
                    frac_var_err = np.sqrt(
                        (1/(2*valid_count*frac_var)) *
                        (mean_err_squared/mean_flux**2) +
                        (2*mean_err_squared**2/(valid_count*mean_flux**4))
                    )
                else:
                    frac_var_err = np.nan
            else:
                frac_var = np.nan
                frac_var_err = np.nan
            
            # Model performance metrics with NaN handling
            try:
                y_pred_norm = gp_model.predict(X_norm)
                y_pred = y_pred_norm * self.y_std + self.y_mean
                
                # Filter out NaN values for metrics calculation
                valid_indices = ~np.isnan(y) & ~np.isnan(y_pred)
                if np.any(valid_indices):
                    mse = mean_squared_error(y[valid_indices], y_pred[valid_indices])
                    r2 = r2_score(y[valid_indices], y_pred[valid_indices])
                else:
                    mse = np.nan
                    r2 = np.nan
                    
                log_likelihood = gp_model.log_marginal_likelihood_value_
                
                # Calculate chi-squared with NaN handling
                residuals = y - y_pred
                if 'e_cts' in data.columns:
                    errors = data['e_cts'].values
                    # Filter out zeros, NaNs, and infinities
                    valid_indices = (errors > 0) & (~np.isnan(residuals)) & (~np.isnan(errors)) & (~np.isinf(residuals)) & (~np.isinf(errors))
                    if np.any(valid_indices):
                        chi2 = np.sum((residuals[valid_indices] / errors[valid_indices]) ** 2)
                        reduced_chi2 = chi2 / max(1, (np.sum(valid_indices) - 2))  # 2 parameters in the model
                    else:
                        chi2 = np.nan
                        reduced_chi2 = np.nan
                else:
                    chi2 = np.nan
                    reduced_chi2 = np.nan
            except Exception as e:
                logging.error(f"Error calculating model metrics: {e}")
                mse = np.nan
                r2 = np.nan
                log_likelihood = np.nan
                chi2 = np.nan
                reduced_chi2 = np.nan

            # Extract kernel parameters
            kernel_params = self.extract_kernel_parameters(gp_model)
            
            # Compile all statistics
            stats = {
                'Data Statistics': {
                    'Number of data points': len(y),
                    'Time span (days)': time_span,
                    'Mean flux (counts)': mean_flux,
                    'Standard deviation (counts)': std_flux,
                    'Min flux (counts)': min_flux,
                    'Max flux (counts)': max_flux,
                    'Amplitude (counts)': amplitude,
                    'RMS': rms,
                    'SNR': snr,
                    'Fractional variability': frac_var,
                    'Fractional variability error': frac_var_err
                },
                'GP Model Statistics': {
                    'Characteristic timescale (days)': kernel_params['tau_drw_days'],
                    'Amplitude (counts)': kernel_params['amplitude'],
                    'Noise level': kernel_params['noise_level'],
                    'Mean Squared Error': mse,
                    'R-squared': r2,
                    'Chi-squared': chi2,
                    'Reduced Chi-squared': reduced_chi2,
                    'Log-likelihood': log_likelihood
                }
            }
            return stats
        except Exception as e:
            logging.error(f"Error in calculate_statistics: {e}")
            # Return a dictionary with NaN values for all statistics
            return {
                'Data Statistics': {
                    'Number of data points': 0,
                    'Time span (days)': np.nan,
                    'Mean flux (counts)': np.nan,
                    'Standard deviation (counts)': np.nan,
                    'Min flux (counts)': np.nan,
                    'Max flux (counts)': np.nan,
                    'Amplitude (counts)': np.nan,
                    'RMS': np.nan,
                    'SNR': np.nan,
                    'Fractional variability': np.nan,
                    'Fractional variability error': np.nan
                },
                'GP Model Statistics': {
                    'Characteristic timescale (days)': np.nan,
                    'Amplitude (counts)': np.nan,
                    'Noise level': np.nan,
                    'Mean Squared Error': np.nan,
                    'R-squared': np.nan,
                    'Chi-squared': np.nan,
                    'Reduced Chi-squared': np.nan,
                    'Log-likelihood': np.nan
                }
            }

    def predict_gp(self, gp, X_norm, X_pred=None, n_samples=20):
        """Make predictions with the GP model"""
        try:
            if X_pred is None:
                # Create a denser grid for predictions
                X_min, X_max = X_norm.min(), X_norm.max()
                X_pred_norm = np.linspace(X_min, X_max, 1000).reshape(-1, 1)
            else:
                # Normalize the provided prediction points
                X_pred_norm = (X_pred - self.X_mean) / self.X_std
            
            # Predict mean and standard deviation
            y_pred_norm, sigma_norm = gp.predict(X_pred_norm, return_std=True)
            
            # Denormalize
            X_pred = X_pred_norm * self.X_std + self.X_mean
            y_pred = y_pred_norm * self.y_std + self.y_mean
            sigma = sigma_norm * self.y_std
            
            # Generate samples from the posterior
            samples_norm = gp.sample_y(X_pred_norm, n_samples=n_samples)
            samples = samples_norm * self.y_std + self.y_mean
            
            return X_pred, y_pred, sigma, samples
        except Exception as e:
            logging.error(f"Error in predict_gp: {e}")
            return None, None, None, None

    def plot_gp_results(self, data, X_pred, y_pred, sigma, samples, obj_name, stats, agn_class=None, save_dir=None):
        """Plot the GP regression results with enhanced GP line and statistics"""
        try:
            plt.figure(figsize=(14, 8))
            
            # Plot the original data with error bars
            plt.errorbar(data['BTJD'], data['cts'], yerr=data['e_cts'],
                        fmt='o', color='black', ecolor='lightgray',
                        elinewidth=1, capsize=2, markersize=4, alpha=0.5, label='Observations')
            
            # Plot the GP mean prediction with enhanced visibility
            plt.plot(X_pred, y_pred, 'r-', linewidth=2.5, label='GP Mean')
            
            # Plot the confidence interval
            plt.fill_between(X_pred.flatten(),
                            y_pred - 2*sigma,
                            y_pred + 2*sigma,
                            color='red', alpha=0.2, label='95% Confidence Interval')
            
            # Plot a few samples from the posterior
            if samples is not None:
                for i in range(min(5, samples.shape[1])):
                    plt.plot(X_pred, samples[:, i], 'r-', alpha=0.1)
            
            # Add sector information
            sectors = data['sector'].unique()
            sectors.sort()
            for sector in sectors:
                sector_data = data[data['sector'] == sector]
                mid_point = len(sector_data) // 2
                if not sector_data.empty and mid_point < len(sector_data):
                    x_pos = sector_data.iloc[mid_point]['BTJD']
                    y_max = sector_data['cts'].max()
                    plt.text(x_pos, y_max * 1.05, f"S{sector}",
                            horizontalalignment='center', fontsize=8)
            
            # Add AGN class to title if provided
            title = f"Gaussian Process Regression for {obj_name}"
            if agn_class:
                title += f" ({agn_class})"
            plt.title(title, fontsize=14)
            
            plt.xlabel('BTJD (Barycentric TESS Julian Date)', fontsize=12)
            plt.ylabel('Counts (cts)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Add key statistics to the plot
            # Handle potential NaN values in statistics
            tau = stats['GP Model Statistics']['Characteristic timescale (days)']
            tau_str = f"{tau:.2f}" if not np.isnan(tau) else "N/A"
            
            amp = stats['GP Model Statistics']['Amplitude (counts)']
            amp_str = f"{amp:.2f}" if not np.isnan(amp) else "N/A"
            
            fvar = stats['Data Statistics']['Fractional variability']
            fvar_str = f"{fvar:.4f}" if not np.isnan(fvar) else "N/A"
            
            r2 = stats['GP Model Statistics']['R-squared']
            r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
            
            rms = stats['Data Statistics']['RMS']
            rms_str = f"{rms:.4f}" if not np.isnan(rms) else "N/A"
            
            snr = stats['Data Statistics']['SNR']
            snr_str = f"{snr:.4f}" if not np.isnan(snr) else "N/A"
            
            red_chi2 = stats['GP Model Statistics']['Reduced Chi-squared']
            red_chi2_str = f"{red_chi2:.4f}" if not np.isnan(red_chi2) else "N/A"
            
            stats_text = (
                f"Characteristic timescale: {tau_str} days\n"
                f"Amplitude: {amp_str} counts\n"
                f"Fractional variability: {fvar_str}\n"
                f"R-squared: {r2_str}\n"
                f"RMS: {rms_str}\n"
                f"SNR: {snr_str}\n"
                f"Reduced χ²: {red_chi2_str}"
            )
            
            # Position the stats box in the upper left corner
            plt.figtext(0.15, 0.85, stats_text,
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                       fontsize=10)
            
            # Add sector information at the bottom
            plt.figtext(0.5, 0.01, f"Sectors: {', '.join([f'S{s}' for s in sectors])}",
                      ha="center", fontsize=8, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for bottom text
            
            if save_dir:
                # Create subdirectory for AGN class if provided
                if agn_class:
                    save_dir = os.path.join(save_dir, agn_class)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{obj_name}_gp_regression.png")
                plt.savefig(save_path, dpi=300)
                logging.info(f"GP regression plot saved to {save_path}")
            
            plt.close()
        except Exception as e:
            logging.error(f"Error in plot_gp_results: {e}")

    def save_statistics_to_csv(self, obj_name, stats, sectors, agn_class=None, save_dir=None):
        """Save the statistics to a CSV file with proper NaN handling"""
        try:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                csv_path = os.path.join(save_dir, "light_curve_statistics.csv")
                
                # Check if file exists to determine if we need to write headers
                file_exists = os.path.isfile(csv_path)
                
                # Prepare the data row
                data_row = {
                    'Object Name': obj_name,
                    'AGN Class': agn_class if agn_class else "Unknown",
                    'Characteristic Timescale (days)': stats['GP Model Statistics']['Characteristic timescale (days)'],
                    'Fractional Variability': stats['Data Statistics']['Fractional variability'],
                    'R-squared': stats['GP Model Statistics']['R-squared'],
                    'RMS': stats['Data Statistics']['RMS'],
                    'SNR': stats['Data Statistics']['SNR'],
                    'Sectors': ','.join(map(str, sectors)),
                    'Amplitude (counts)': stats['Data Statistics']['Amplitude (counts)'],
                    'Chi-squared': stats['GP Model Statistics']['Chi-squared'],
                    'Reduced Chi-squared': stats['GP Model Statistics']['Reduced Chi-squared'],
                    'Number of data points': stats['Data Statistics']['Number of data points'],
                    'Time span (days)': stats['Data Statistics']['Time span (days)'],
                    'Mean flux (counts)': stats['Data Statistics']['Mean flux (counts)'],
                    'Standard deviation (counts)': stats['Data Statistics']['Standard deviation (counts)'],
                    'Noise level': stats['GP Model Statistics']['Noise level'],
                    'Log-likelihood': stats['GP Model Statistics']['Log-likelihood']
                }
                
                # Replace NaN values with "NA" string
                for key, value in data_row.items():
                    if isinstance(value, (float, np.float64, np.float32)) and (np.isnan(value) or np.isinf(value)):
                        data_row[key] = "NA"
                
                # Write to CSV
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(data_row.keys()))
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(data_row)
                
                logging.info(f"Statistics for {obj_name} saved to {csv_path}")
        except Exception as e:
            logging.error(f"Error in save_statistics_to_csv: {e}")

    def run_full_analysis(self, obj_name, agn_class=None, save_dir=None):
        """Run a complete analysis pipeline for a given object"""
        try:
            # 1. Load and combine data
            logging.info(f"Starting analysis for {obj_name}")
            data = self.combine_sector_data(obj_name, agn_class)
            if data is None or data.empty:
                logging.error(f"No data found for {obj_name}")
                return None
            logging.info(f"Found {len(data)} data points across {len(data['sector'].unique())} sectors")
            
            # 2. Fit GP model
            logging.info("Fitting GP model")
            gp_model, X_norm, y_norm, y_err_norm = self.fit_gp_model(data)
            if gp_model is None:
                logging.error(f"Failed to fit GP model for {obj_name}")
                return None
            
            # 3. Calculate statistics
            stats = self.calculate_statistics(data, gp_model, X_norm, y_norm)
            
            # 4. Make predictions
            X_pred, y_pred, sigma, samples = self.predict_gp(gp_model, X_norm)
            if X_pred is None:
                logging.error(f"Failed to make predictions for {obj_name}")
                return None
            
            # 5. Plot results
            self.plot_gp_results(data, X_pred, y_pred, sigma, samples, obj_name, stats, agn_class, save_dir)
            
            if save_dir:
                sectors = sorted(data['sector'].unique())
                self.save_statistics_to_csv(obj_name, stats, sectors, agn_class, save_dir)
            
            return {
                'data': data,
                'gp_model': gp_model,
                'X_norm': X_norm,
                'y_norm': y_norm,
                'statistics': stats,
                'predictions': {
                    'X_pred': X_pred,
                    'y_pred': y_pred,
                    'sigma': sigma,
                    'samples': samples
                }
            }
        except Exception as e:
            logging.error(f"Error in run_full_analysis for {obj_name}: {e}")
            return None

def get_agn_objects(root_directory, agn_classes):
    """Get all AGN objects from the catalogs"""
    objects_by_class = defaultdict(list)
    sectors = ['02', '03', '04', '05', '07', '19', '20', '21']
    
    # Find all objects in all sectors
    for sector in sectors:
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
                for agn_class in agn_classes:
                    mask = cat.agnclass == agn_class
                    if any(mask):
                        for obj_name in cat.objname[mask]:
                            if obj_name not in objects_by_class[agn_class]:
                                objects_by_class[agn_class].append(obj_name)
                                logging.info(f"Found {agn_class} object: {obj_name}")
            except Exception as e:
                logging.error(f"Error processing catalog {catalog_file}: {e}")
    
    return objects_by_class

def read_object_list(file_path):
    """Read a list of objects from a text file, one object per line."""
    try:
        with open(file_path, 'r') as f:
            # Strip whitespace and filter out empty lines
            objects = [line.strip() for line in f if line.strip()]
        logging.info(f"Read {len(objects)} objects from {file_path}")
        return objects
    except Exception as e:
        logging.error(f"Error reading object list from {file_path}: {e}")
        return []

def main():
    # Set the root directory
    root_directory = '/home/kicowlin/SummerResearch2024'
    save_directory = f'{root_directory}/gp_regression'
    
    # Create the GPR object
    gpr = LightCurveGPR(root_directory)
    
    # Option 1: Process objects from a list file
    obj_list_file = f'{root_directory}/object_list.txt'
    if os.path.exists(obj_list_file):
        obj_list = read_object_list(obj_list_file)
        # Process each object
        for obj_name in obj_list:
            logging.info(f"Processing {obj_name}")
            gpr.run_full_analysis(obj_name, save_dir=save_directory)
    
    # Option 2: Process AGN objects by class
    else:
        # Get all AGN objects by class
        objects_by_class = get_agn_objects(root_directory, gpr.agn_classes)
        
        # Process each object by class
        for agn_class, obj_names in objects_by_class.items():
            logging.info(f"Processing {len(obj_names)} objects for class {agn_class}")
            class_save_dir = os.path.join(save_directory, agn_class)
            os.makedirs(class_save_dir, exist_ok=True)
            
            for obj_name in obj_names:
                logging.info(f"Processing {obj_name} ({agn_class})")
                gpr.run_full_analysis(obj_name, agn_class=agn_class, save_dir=class_save_dir)

if __name__ == "__main__":
    main()
