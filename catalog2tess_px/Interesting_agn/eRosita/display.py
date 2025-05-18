
from astropy.io import fits
import sys

def display_info(filename):
    print(f"Reading file: {filename}")
    print("_" * 50)

    with fits.open(filename) as hdul:
        print("File format:")
        hdul.info()
        data = hdul[1].data
        print("\nColumns:")
        for col in data.columns:
            print(f"- {col.name}: {col.format}")
        print("\nFirst 5 rows of data:")
        for row in data[:5]:
            print(row)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python display.py <fits_file1> [fits_file2 ...]")
    else: 
        for fits_file in sys.argv[1:]:
            display_info(fits_file)

