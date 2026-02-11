from multiprocessing import Pool
import glob
import time
from astropy.time import Time

def append_file_name(file):
    # charger, traiter, sauvegarder
    print(f"{Time.now()} Processing file: {file}")
    time.sleep(15)

if __name__ == '__main__':
    
    files = glob.glob('/home/eartigau/projects/rrg-rdoyon/eartigau/tapas/test_fit/hotstars_NIRPS/*.fits')[0:64]
    print(f'Found {len(files)} files to process.') 
    with Pool(processes=8) as pool:
        pool.map(append_file_name, files)