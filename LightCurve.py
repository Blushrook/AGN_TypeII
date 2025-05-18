import pandas as pd
import matplotlib.pyplot as plt
import os

class LightCurveData:
    def __init__(self, directory, save_directory):
        self.directory = directory
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)

    def load_data(self, filename):
        file_path = os.path.join(self.directory, filename)
        try:
            data = pd.read_csv(file_path, sep=r'\s+')  
            data.columns = data.columns.str.strip() 
            data = data.replace('-', pd.NA).astype(float)  
            print(f"File {filename} loaded successfully")
            return data
        except FileNotFoundError as e:
            print(f"Failed to find the file {filename}:", e)
        except Exception as e:
            print(f"An error occurred while reading {filename}:", e)
        return None

    def plot_light_curve(self, data, title='Light Curve', filename='plot.png'):
        if data is not None and not data.empty:
            try:
                plt.figure(figsize=(10, 6))
                plt.errorbar(data['BTJD'], data['cts'], yerr=data['e_cts'], fmt='o', color='blue', ecolor='lightgray', elinewidth=3, capsize=0)
                plt.title(title)
                plt.xlabel('BTJD (Barycentric TESS Julian Date)')
                plt.ylabel('Counts (cts)')
                plt.grid(True)
                save_path = os.path.join(self.save_directory, filename)
                plt.savefig(save_path)
                plt.close()
                print(f"Plot saved to {save_path}")
            except KeyError as e:
                print(f"Key error: {e} - Check that 'BTJD', 'cts', and 'e_cts' are in your DataFrame")
            except Exception as e:
                print(f"An error occurred while plotting {title}: {e}")
        else:
            print("No data to plot.")

    def process_all_files(self):
        files = [f for f in os.listdir(self.directory) if f.startswith('lc_')]
        for file in files:
            data = self.load_data(file)
            if data is not None:
                plot_filename = f"{os.path.splitext(file)[0]}.png" 
                self.plot_light_curve(data, title=f'Light Curve for {file}', filename=plot_filename)
directory = '/home/kicowlin/SummerResearch2024/Sector07/cam4_ccd4/lc_hyperleda'
save_directory = '/home/kicowlin/SummerResearch2024/lc_plots/Sector07/cam4_ccd4'
light_curve_manager = LightCurveData(directory, save_directory)

light_curve_manager.process_all_files()
