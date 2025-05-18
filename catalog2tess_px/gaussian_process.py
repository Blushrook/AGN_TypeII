import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LightCurveGPR:
    def __init__(self, root_directory):
        self.root_directory = root_directory
        
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
    
    def combine_sector_data(self, obj_name):
        """Combine data for an object across all sectors"""
        combined_data = []
        sectors = ['02', '03', '04', '05', '07', '19', '20', '21']
        
        # Search for the object in all sectors, cameras, and CCDs
        for sector in sectors:
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
        # Extract time and flux data
        X = data['BTJD'].values.reshape(-1, 1)
        y = data['cts'].values
        y_err = data['e_cts'].values
        
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
        drw_kernel = amplitude * Matern(length_scale=length_scale, length_scale_bounds=(1.0, 100.0), nu=0.5)
        
        # Add white noise kernel to account for measurement errors
        noise_kernel = WhiteKernel(noise_level=np.mean(y_err_norm**2), 
                                  noise_level_bounds=(np.min(y_err_norm**2), np.max(y_err_norm**2)*10))
        
        # Combined kernel
        kernel = drw_kernel + noise_kernel
        
        # Create and fit the GP model
        gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_norm**2,
                                     n_restarts_optimizer=n_restarts, normalize_y=False)
        gp.fit(X_norm, y_norm)
        
        logging.info(f"Optimized kernel: {gp.kernel_}")
        
        return gp, X_norm, y_norm, y_err_norm
    
    def predict_gp(self, gp, X_norm, X_pred=None, n_samples=20):
        """Make predictions with the GP model"""
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

    def calculate_rms(self, data):
        try:
            if data is None or data.empty:
                logging.warning("Cannont calculate RMS:data is empty or None")
                return None
            
            mean_cts = data['cts'].mean()
            square_diff = (data['cts'] - mean_cts) ** 2
            mean_square_diff = squared_deff.mean()
            rms = np.sqrt(mean_squared_deff)
            
            logging.info(f"Light curve RMS: {rms:.4f}")
            return rms
        except Exception as e:
            logging.error(f"An error occured while calculating RMS: {e}")
            return None

    def calculate_snr(self,data):
        try:
            if data is None or data.empty:
                logging.warning("Cannot calculate SNR: data is empty or None")
                return None
            signal = data['cts'}.mean()
            snr = signal / noise if noise > 0 else float('inf')

            logging.info(f"Light curve SNR: {snr:.4f}")
            return snr
        except Exception as e:
            logging.error(f"An error occurred wile calculating SNR: {e}")
            return None

    def plot_gp_results(self, data, X_pred, y_pred, sigma, samples, obj_name, save_dir=None):
        """Plot the GP regression results"""
        plt.figure(figsize=(14, 8))
        
        # Plot the original data with error bars
        plt.errorbar(data['BTJD'], data['cts'], yerr=data['e_cts'], 
                    fmt='o', color='black', ecolor='lightgray',
                    elinewidth=1, capsize=2, markersize=4, label='Observations')
        
        # Plot the GP mean prediction
        plt.plot(X_pred, y_pred, 'r-', label='GP Mean')
        
        # Plot the confidence interval
        plt.fill_between(X_pred.flatten(), 
                        y_pred - 2*sigma, 
                        y_pred + 2*sigma, 
                        color='red', alpha=0.2, label='95% Confidence Interval')
        
        # Plot a few samples from the posterior
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
        
        plt.title(f"Gaussian Process Regression for {obj_name}")
        plt.xlabel('BTJD (Barycentric TESS Julian Date)')
        plt.ylabel('Counts (cts)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        info_text = f"Sectors: {', '.join([f'S{s}' for s in sectors]_}"
        if rms in not None:
            info_text: += f" | RMS: {rms:.4f}"
        if snr is not None:
            info_text: += f" | SNR: {snr:.4f}"

        # Add kernel information
        plt.figtext(0.5, 0.01, info_text,
                  ha="center", fontsize=8, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{obj_name}_gp_regression.png")
            plt.savefig(save_path, dpi=300)
            logging.info(f"GP regression plot saved to {save_path}")
        
        plt.show()

def main():
    # Set the root directory
    root_directory = '/home/kicowlin/SummerResearch2024'
    save_directory = f'{root_directory}/gp_regression'
    
    # Create the GPR object
    gpr = LightCurveGPR(root_directory)
    
    # Process NGC1566
    obj_name = 'NGC1566'
    logging.info(f"Processing {obj_name}")
    
    # Load and combine data
    combined_data = gpr.combine_sector_data(obj_name)
    
    if combined_data is not None and len(combined_data) > 0:
        rms = gpr.combine_sector_data(combined_data)
        snr = grp.calculate_snr(combined_data)

        # Fit the GP model
        gp_model, X_norm, y_norm, y_err_norm = gpr.fit_gp_model(combined_data)
        
        # Make predictions
        X_pred, y_pred, sigma, samples = gpr.predict_gp(gp_model, X_norm)
        
        # Plot the results
        gpr.plot_gp_results(combined_data, X_pred, y_pred, sigma, samples, obj_name, save_directory)
        
        logging.infor(f"Object: {obj_name}, RMS: {rms: .4f}, SNR: {snr:.4f}")

    else:
        logging.error(f"No data available for {obj_name}")

if __name__ == "__main__":
    main()

