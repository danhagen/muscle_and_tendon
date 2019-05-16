import numpy as np
import os.path
import time
import matplotlib._pylab_helpers
from matplotlib.backends.backend_pdf import PdfPages
import plotly.plotly as py
import plotly.tools as tls

def return_length_of_nonzero_array(X):
	"""
	Takes in a numpy.ndarray X of shape (m,n) and returns the length of the array that removes any trailing zeros.
	"""
	assert str(type(X))=="<class 'numpy.ndarray'>", "X should be a numpy array"
	assert np.shape(X)[1]!=1, "X should be a wide rectangular array. (m,1) is a column, therefore a nonzero X of this shape will return 1 (trivial solution). Transpose X to properly identify nonzero array length."
	assert np.shape(X)!=(1,1), "Check input. Should not be of shape (1,1) (trivial solution)."
	if (X[:,1:]!=np.zeros(np.shape(X[:,1:]))).all():
		return(np.shape(X)[1])
	else:
		return(np.argmax((X[:,1:] == np.zeros(np.shape(X[:,1:]))).sum(axis=0) == np.shape(X[:,1:])[0])+1)

def save_figures(Destination,BaseFileName,**kwargs):
	"""

	"""
	ReturnPath = kwargs.get("ReturnPath",False)
	assert type(ReturnPath)==bool, "ReturnPath must be either true or false (default)."
	SubFolder = kwargs.get("SubFolder",time.strftime("%Y_%m_%d_%H%M%S")+"/")
	FilePath = Destination + SubFolder
	assert type(Destination) == str and Destination[-1] == "/", \
		"Destination must be a string ending is '/'. Currently Destination = " + str(Destination)
	assert type(SubFolder) == str and SubFolder[-1] == "/", \
		"SubFolder must be a string ending is '/'. Currently SubFolder = " + str(SubFolder)

	if not os.path.exists(FilePath):
		os.makedirs(FilePath)

	figs = kwargs.get("figs",
		[manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
		)

	SaveAsPDF = kwargs.get("SaveAsPDF",False)
	assert type(SaveAsPDF)==bool, "SaveAsPDF must be either True or False."

	i = 1
	FileName = BaseFileName + "_" + "{:0>2d}".format(i) + "-01.jpg"
	if os.path.exists(FilePath + FileName) == True:
		while os.path.exists(FilePath + FileName) == True:
			i += 1
			FileName = BaseFileName + "_" + "{:0>2d}".format(i) + "-01.jpg"

	for i in range(len(figs)):
		figs[i].savefig(FilePath + FileName[:-6] + "{:0>2d}".format(i+1) + ".jpg")

	if SaveAsPDF == True:
		PDFFileName = FileName[:-7] + ".pdf"
		assert not os.path.exists(FilePath + PDFFileName), \
				("Error with naming file. "
				+ PDFFileName
				+ " should not already exist as "
				+ FileName
				+ " does not exist. Try renaming or deleting "
				+ PDFFileName
				)

		PDFFile = PdfPages(FilePath + PDFFileName)
		if len(figs)==1:
			PDFFile.savefig(figs[0])
		else:
			[PDFFile.savefig(fig) for fig in figs]
		PDFFile.close()
	if ReturnPath==True:
		return(FilePath)

def save_figures_to_plotly(FileName,**kwargs):
	"""

	"""
	figs = kwargs.get("figs",
		[manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
		)

	FileTime = time.strftime("%Y_%m_%d_%H%M%S")
	for i in range(len(figs)):
		plotly_fig = tls.mpl_to_plotly(figs[i])
		py.plot(plotly_fig,filename=(FileName + "-" + FileTime + "-" + "{:0>2d}".format(i+1)))
