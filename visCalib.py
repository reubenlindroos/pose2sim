import numpy as np
import sys
import matplotlib.pyplot as plt

import toml

from scipy.spatial.transform import Rotation as spR
import os
import argparse


class Calib:
	def __init__(self):
		self.xres = 0
		self.yres = 0
		self.K = np.eye(3,3)
		self.L = np.eye(4,4)
		self.k = [0,0,0,0,0]
		
		
	def FromCalib(self, fname):
		"""Load from .calib file"""
		infi = open(fname)
		lines = infi.readlines()
		
		# Note: stupidly assuming that the calib file has the blank lines
		#       it always does but the C++ code doesn't assume it.
		
		self.xres = float( lines[0] )
		self.yres = float( lines[1] )
		self.K    = np.array( [ [float(s) for s in l.split()] for l in lines[2:5] ] )
		
		self.L    = np.array( [ [float(s) for s in l.split()] for l in lines[6:10] ] )
		
		self.k    = np.array( [ float(s) for s in lines[11].split() ] )
	
	def FromToml(self, tdict, camName):
		"""Load from toml dictionary"""
		cam = tdict[camName]
		
		
		self.xres = cam["size"][0]
		self.yres = cam["size"][1]
		self.K    = cam["matrix"]
		
		Rvec      = cam["rotation"]
		tvec      = cam["translation"]
		
		R0 = spR.from_rotvec( Rvec )
		R1 = R0.as_matrix()
		Li = np.eye(4,4)
		Li[0:3,0:3] = R1
		Li[0:3,  3] = tvec #np.dot(1000,tvec)
		
		self.L = np.linalg.inv( Li )
		
		
		
		self.k    = cam["distortions"]
		
		

def Scatter( pts, colour, ax ):
	x = pts[:,0,:]
	y = pts[:,1,:]
	z = pts[:,2,:]
	ax.scatter3D(x, y, z, color=colour)

def Lines( pts0, pts1, colour, ax ):
	assert( pts0.shape == pts1.shape )
	for pc in range( pts0.shape[0] ):
		x = [ pts0[pc,0,0], pts1[pc,0,0] ]
		y = [ pts0[pc,1,0], pts1[pc,1,0] ]
		z = [ pts0[pc,2,0], pts1[pc,2,0] ]
		ax.plot3D( x, y, z, color=colour )


#
# Load calib files
#

def load_calibs(*args):
	"""
	takescommand line agruments and returns a list of calibs
	"""

	calFiles = []
	if args[0].get("calib_dir") is not None:
		calib_dir_contents = os.listdir(args[0].get("calib_dir"))
		for item in calib_dir_contents:
			if item.endswith(".calib"):
				calFiles.append(os.path.join(args[0].get("calib_dir"),item))

	tmlFiles = []
	if args[0].get("toml_dir") is not None:
		tmlDir = args[0].get("toml_dir")
		for item in os.listdir(tmlDir):
			if item.endswith(".toml"):
				tmlFiles.append(os.path.join(args[0].get("toml_dir"),item))
	calibs = []
	for fn in calFiles:
		c = Calib()
		c.FromCalib( fn )
		calibs.append(c)

	for fn in tmlFiles:
		print(fn)
		tdict = toml.load(fn)
		for cname in tdict:
			if cname.find("cam_") == 0:
				c = Calib()
				c.FromToml( tdict, cname )
				calibs.append(c)
	return calibs

#
# Overly simplistic plotting...
#
def plot(calibs):
	o = np.array([[0, 0, 0, 1]]).T
	d = 1000
	x = np.array([[d, 0, 0, 1]]).T
	y = np.array([[0, d, 0, 1]]).T
	z = np.array([[0, 0, d, 1]]).T
	centres = np.array([np.dot(np.linalg.inv(c.L), o) for c in calibs])
	xs = np.array([np.dot(np.linalg.inv(c.L), x) for c in calibs])
	ys = np.array([np.dot(np.linalg.inv(c.L), y) for c in calibs])
	zs = np.array([np.dot(np.linalg.inv(c.L), z) for c in calibs])

	print(centres.shape, xs.shape, ys.shape, zs.shape)

	#
	# Render stuff
	#

	fig = plt.figure(figsize=(9, 6))
	ax = plt.axes(projection='3d')

	Scatter(centres, 'black', ax)
	Lines(centres, xs, 'red', ax)
	Lines(centres, ys, 'green', ax)
	Lines(centres, zs, 'blue', ax)

	ax.set_title("camera pos vis", pad=25, size=15)

	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")

	ax.axes.set_xlim3d(left=-5000, right=5000)
	ax.axes.set_ylim3d(bottom=-5000, top=5000)
	ax.axes.set_zlim3d(bottom=0, top=10000)

	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-ic', '--calib_dir', required=False, help='Directory where .calib files are located')
	parser.add_argument('-it', '--toml_dir', required=False, help='path to toml file')
	args = vars(parser.parse_args())
	calibs = load_calibs(args)
	plot(calibs)

