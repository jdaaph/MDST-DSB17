from tqdm import *
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os

# Load the scans in given folder path

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def main():
    # Some constants 
    INPUT_FOLDER = '/scratch/mdatascienceteam_flux/shared/DSB_2017/data/sample_images/'
    OUTPUT_FOLDER = '/scratch/mdatascienceteam_flux/shared/dsb_external/sample/'
    patients = os.listdir(INPUT_FOLDER)[1:]
    patients.sort()

    y = []
    for i in tqdm(range(len(patients))):
        p_pixels = get_pixels_hu(load_scan(INPUT_FOLDER + patients[i]))
        if not i:
            total_data = p_pixels
        else:
            total_data = np.vstack([total_data, p_pixels])
        num_slices = p_pixels.shape[0]
        tmp_y = [i / num_slices for i in range(num_slices)]
        y.extend(tmp_y)

    np.save(OUTPUT_FOLDER+'sample_X.npy', total_data)
    np.save(OUTPUT_FOLDER+'sample_y.npy', np.array(y))

def load_sample(seed=111):
    FOLDER = '/scratch/mdatascienceteam_flux/shared/dsb_external/sample/'
    X = np.load(FOLDER + 'sample_X.npy')
    y = np.load(FOLDER + 'sample_y.npy')

    nb_train_samples = X.shape[0]
    x_total = np.zeros((nb_train_samples, 1, 512, 512), dtype='int16')
    y_total = np.zeros((nb_train_samples,), dtype='float')

    x_total[:,0,:,:] = X
    y_total[:] = y

    np.random.seed(seed)
    indices = numpy.random.permutation(nb_train_samples)
    cutoff = np.ceil(0.8*nb_train_samples)
    train_idx, test_idx = indices[:cutoff], indices[cutoff:]

    return (x_total[train_idx,:,:,:], y_total[train_idx]), (x_total[test_idx,:,:,:], y_total[test_idx])

if __name__ == "__main__":
    main()
