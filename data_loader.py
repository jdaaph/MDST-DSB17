import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os

class DataLoader():
	"""
	Version 1 of data loader. 
	"""
	def __init__(self,fpath = '/scratch/mdatascienceteam_flux/shared/DSB_2017/', csv_file = 'stage1_labels.csv'):
		self.fpath = fpath
		self.csv_file = csv_file

	def _load_scan(self, path):
		slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
		slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
		try:
			slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
		except:
			slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
		
		for s in slices:
			s.SliceThickness = slice_thickness
		
		return slices
	def _get_pixels_hu(self, slices):
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

	"""
	Public Interface for loading the data
	"""
	def load_sample_ids(self):
		"""
		@ Input: file path (the folder that contains the csv label file), csv_file name  
		@ returns a list of viable sample ids
		"""
		label_file = pd.read_csv(self.fpath + self.csv_file)
		return label_file['id'].tolist()
	
	def load_example(self, id1):
		"""
		@ Input: id of a sample
		Note: this function may only work in flux, due to the folder arrangement
		@ return 3D image
		"""
		path = self.fpath+'data/stage1/stage1/' + str(id1)
		slices = self._load_scan(path)
		return self._get_pixels_hu(slices)
		
	def load_label(self, id1):
		label_file = pd.read_csv(self.fpath + self.csv_file)
		return label_file.ix[label_file['id'] == id1, 'cancer'].values[0]


if __name__ == '__main__':
	# test
	# default: fpath: path for the csv file; csv_name: the csv file of labels 
	dl = DataLoader()
	# return all the ids from file: stage1_labels.csv
	ids = dl.load_sample_ids()
	print ids[:10]
	print 'number of patients', len(ids)
	# load 3D image for the first patient
	p1 = dl.load_example(ids[0])
	print p1.shape, type(p1)
	# print the first patient label (cancer:1, not cancer: 0)
	print dl.load_label(ids[0])
