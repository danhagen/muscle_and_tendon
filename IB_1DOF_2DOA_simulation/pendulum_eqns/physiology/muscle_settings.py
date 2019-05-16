import numpy as np
from numpy import pi
from collections import namedtuple

def return_muscle_settings(PreselectedMuscles=None):
	"""
	Notes:
	Coefficients from observation, Ramsay; 2009, FVC, Holtzbaur, Pigeon, Kuechle, or Banks. Optimal Muscle Length given in mm. Optimal tendon/muscle lengths and PCSA were taken from Garner and Pandy (2003)
	"""

	# Coefficients from observation, Ramsay; 2009, Pigeon, FVC, Holtzbaur, Garner & Pandy, or Banks.

	MA_Settings = namedtuple("MA_Settings",["Values","Source","Units","Equation_Number","Threshold","DOF"])
	Spindle_Settings = namedtuple("Spindle_Settings",["ActualNumber",'CorrectedNumber','RelativeAbundance',"Source"])
	Input_Source = namedtuple("Source_Settings",["Values","Source","Units"])

	def Pigeon_coeff_conversion(Coefficients):
		"""
		Takes in Coefficient values from Pigeon (1996) -- which take in angles in degrees -- and coverts them into the properly scaled coefficients for radians, additionally scaled by the magnitude listed in the paper.

		Note that the coefficients listed in Pigeon (1996) are given in decending order (i.e., c₅,c₄,c₃,c₂,c₁,c₀). However to maintain continuity with the equations given in Ramsay; 2009 (2009), we list coefficients in order of increasing power (i.e., c₀,c₁,c₂,c₃,c₄,c₅).
		"""
		import numpy as np
		assert len(Coefficients)==6, 'For Pigeon (1996) the list of Coefficients must be 6 elements long. Insert zeros (0) for any additional empty coefficients.'
		assert type(Coefficients)==list, 'Coefficients must be a 6 element list.'
		Rad_Conversion = np.multiply(Coefficients,\
				np.array([1,(180/np.pi),(180/np.pi)**2,(180/np.pi)**3,(180/np.pi)**4,(180/np.pi)**5],dtype = 'float64'))
		new_Coefficients =\
			np.multiply(Rad_Conversion,np.array([1,1e-1,1e-3,1e-5,1e-7,1e-9],dtype='float64'))
		return(new_Coefficients)

	PC_Settings = {\
		'Notes' : [\
						'This is the *clavicular* portion of the pectoralis major.',\
						'Banks and Garner & Pandy are parameter values for the entire muscle. Pigeon and Holzbaur have the values for the clavicular portion only.',\
						'Holzbaur parameters for shoulder are for frontal plane (ABD/ADD) only! This explains the relatively small values for some measures.'\
					],\
		'Shoulder MA' : {	"Primary Source" : "Pigeon; 1996",\
		 					"Sources" : \
								[\
									MA_Settings([50.80,0,0,0,0,0], 'Pigeon; 1996', "mm", None, None, "Shoulder"),\
									MA_Settings([2,0,0,0,0,0], 'Holzbaur; 2005', "mm", None, None, "Shoulder")\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Est", \
		 				"Sources" : \
							[\
								MA_Settings(0, "m", None, None, 'Elbow', "Est")\
							]}, \
		'Spindle' : Spindle_Settings(450,389.7,1.2,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006", \
					"Sources" : \
						[\
							Input_Source(295.6, "Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(14.4, 'Holzbaur; 2005', 'cm'),\
											Input_Source(150, 'Est', 'mm')\
										]}, \
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(0.3, 'Holzbaur; 2005', 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(17, 'Holzbaur; 2005', 'degrees')\
									]}, \
		'PCSA' : {	"Primary Source" : "Garner & Pandy; 2003", \
					'Sources' : \
						[\
							Input_Source(36.20,'Garner & Pandy; 2003','sq cm'),\
							Input_Source(2.6,'Holzbaur; 2005','sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Garner & Pandy; 2003", \
										'Sources': \
											[\
												Input_Source(1175.01,'Garner & Pandy; 2003','N'),\
												Input_Source(364.4,'Holzbaur; 2005','N')\
											]}\
		}

	DELTa_Settings = {\
		'Notes' : [\
					"SFE MA is listed as 33.02 mm in Pigeon and estimated as 19 mm. Using Pigeon Coefficients convention, Kuechle (1997) has the DELTp MA for [-140,90] as Pigeon_coeff_conversion([ 13.4293189,  2.0316226, -0.2339031,  2.7807828,  0.,  0.]). This will yield a piecewise function that creates jumps with the new velocity formulation. Instead, we are going to try Pigeon_coeff_conversion([ 12.7928795,  2.0480346,  0.8917734,  3.2207214, -2.3928223,  0.]) so that the function is within range and continuous during the ROM. Threshold (pi/2) has been removed for this new MA function.",\
					"Garner & Pandy have much larger PCSA and Peak Force Values but only consider the entire Deltoid.",\
					"Holzbaur parameters for shoulder are for frontal plane (ABD/ADD) only! This explains the relatively small values for some measures.",\
					"Banks only had mass and spindle settings for the entire deltoid. As a result the parameters are divided by 3 as an estimate of the individual muscles. Will need to do sensitivity analysis as a result."\
					], \
		'Shoulder MA' : {	"Primary Source" : "Kuechle; 1997",\
		 					"Sources" : \
								[\
									MA_Settings(Pigeon_coeff_conversion([12.7928795,  2.0480346,  0.8917734,  3.2207214, -2.3928223,  0.]), 'Kuechle; 1997', "mm", None, None, 'Shoulder'),\
									MA_Settings([33.02,0,0,0,0,0], 'Pigeon; 1996', "mm", None, None, 'Shoulder'),\
									MA_Settings([1.9,0,0,0,0,0], 'Holzbaur; 2005', "cm", None, None, 'Shoulder'),\
									MA_Settings(19, "Est", "mm", None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Est", \
		 				"Sources" : \
							[\
								MA_Settings(0, "Est", "m", None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(182/3,426.3/3,0.43,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006", \
					"Sources" : \
						[\
							Input_Source(355.7/3, "Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(9.8, 'Holzbaur; 2005', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(9.3, 'Holzbaur; 2005', 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(22, 'Holzbaur; 2005', 'degrees')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(82.98,'Garner & Pandy; 2003','sq cm'),\
							Input_Source(8.2,'Holzbaur; 2005','sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(2044.65,'Garner & Pandy; 2003','N'),\
												Input_Source(1142.6,'Holzbaur; 2005','N')\
											]}\
		}

	CB_Settings = {\
		'Notes' : [\
						"Holzbaur parameters for shoulder are for frontal plane (ABD/ADD) only! This explains the relatively small values for some measures. MA is negative as a result.",\
						"Garner & Pandy values for muscle length, PCSA, and peak force are very different from those reported in Wood (1989), Veeger (1991), Bassett (1990), Chen (1988), Keating (1993), Veeger (1997), An (1981), and Cutts (1991)." \
					],\
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(20, "Est", "mm", None, None, "Shoulder"),\
									MA_Settings([-20,0,0,0,0,0], "Holzbaur; 2005", "mm", None, None, "Shoulder")\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Est", \
		 				"Sources" : \
							[\
								MA_Settings(0, "Est", "m", None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(123,147.3,0.83,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : 'Banks; 2006',\
					'Sources' : \
						[\
							Input_Source(39.8, "Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(9.3, 'Holzbaur; 2005', 'cm'),\
											Input_Source(17.60, 'Garner & Pandy; 2003', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : 'Holzbaur; 2005', \
									"Sources" : \
										[\
											Input_Source(9.7, 'Holzbaur; 2005', 'cm'),\
											Input_Source(4.23, 'Garner & Pandy; 2003', 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(27, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(1.7,"Holzbaur; 2005","sq cm"),\
							Input_Source(4.55,"Garner & Pandy; 2003","sq cm")\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(242.5, "Holzbaur; 2005", "N"),\
												Input_Source(150.02, "Garner & Pandy; 2003", "N")\
											]}\
		}

	DELTp_Settings = {\
		'Notes' : [\
						"DELTp SFE MA is listed as -78.74 mm in Pigeon. Using Pigeon Coefficients convention, Kuechle (1997) has the DELTp MA for [-140,90] as Pigeon_coeff_conversion([ 22.8547177,  3.9721238, -3.3900829, -3.6146546,  0.,  0.]). This will yield a piecewise function that creates jumps with the new velocity formulation. Instead, we are going to try Pigeon_coeff_conversion([-23.8165173, -4.486164 ,  5.8655808,  6.5003255, -8.2736695,2.0812998]) so that the function is within range and continuous during the ROM. Threshold (pi/2) has been removed for this new MA function.",\
						"Holzbaur parameters for shoulder are for frontal plane (ABD/ADD) only! This explains the relatively small values for some measures. MA is negative as a result.",\
						"Garner & Pandy values for muscle length, PCSA, and peak force are very different from those reported in Wood (1989), Veeger (1991), Bassett (1990), Chen (1988), Keating (1993), Veeger (1997), An (1981), and Cutts (1991). Also, they do no distinguish between ant, mid, post.",\
						"Holzbaur parameters for shoulder are for frontal plane (ABD/ADD) only! This explains the relatively small values for some measures.",\
						"Banks only had mass and spindle settings for the entire deltoid. As a result the parameters are divided by 3 as an estimate of the individual muscles. Will need to do sensitivity analysis as a result."\
					],\
		'Shoulder MA' : {	"Primary Source" : "Kuechle; 1997",\
		 					"Sources" : \
								[\
									MA_Settings(Pigeon_coeff_conversion([-23.8165173, -4.486164 ,  5.8655808,  6.5003255, -8.2736695,2.0812998]), 'Kuechle; 1997', "mm", None, None, "Shoulder"),\
									MA_Settings([-78.74,0,0,0,0,0], 'Pigeon; 1996', "mm", None, None, "Shoulder"),\
									MA_Settings([-8,0,0,0,0,0], 'Holzbaur; 2005', "mm", None, None, "Shoulder")
								]}, \
		'Elbow MA' : {	"Primary Source" : "Est", \
		 				"Sources" : \
							[\
								MA_Settings(0, "Est", "m", None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(182/3,426.3/3,0.43,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006", \
					"Sources" : \
						[\
							Input_Source(355.7/3, "Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(13.7, 'Holzbaur; 2005', 'cm'),\
											Input_Source(12.8, 'Garner & Pandy; 2003', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(3.8, "Holzbaur; 2005", "cm"),\
											Input_Source(5.38, 'Garner & Pandy; 2003', 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(18, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(1.9, "Holzbaur; 2005", 'sq cm'),\
							Input_Source(81.98,"Garner & Pandy; 2003","sq cm")\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(259.9,"Holzbaur; 2005","N"),\
												Input_Source(2044.65,"Garner & Pandy; 2003","N")\
											]}\
		}

	BIC_Settings = {\
		'Notes' : [\
					"BIC EFE MA for Ramsay; 2009 has R² = 0.985 whereas Pigeon has R² = 0.9918. Pigeon, however, only takes elbow angle into account, whereas Ramsay; 2009 takes in variable PS angles. It appears that because Pigeon uses an average of fully pronated and fully supinated MAs, the BIC moment arm is similar but subject to variation as the level of PS is changed. (NOTE: BIC becomes slightly negative when q2 > 3.021. If trajectory has elbow angles exceding this value, enter a threshold of 3.021 into the model.)",\
					"Note: Only using the long head for optimal length, see Holzbaur (2005) for additional head parameters. Adding when logical."
					],\
		'Shoulder MA' : {	"Primary Source" : "Pigeon; 1996",\
		 					"Sources" : \
								[\
									MA_Settings([29.21,0,0,0,0,0], 'Pigeon; 1996', "mm", None, None, "Shoulder"),\
									MA_Settings(15, 'Est', "mm", None, None, "Shoulder")\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Ramsay; 2009", \
	 					"Sources" : \
							[\
								MA_Settings([8.4533,36.6147,2.4777,-19.432,2.0571,0,13.6502,0,0,-5.6172,0,-2.0854,0,0,0,0], 'Ramsay; 2009', "mm", 2, 3.021, 'Elbow'),\
								MA_Settings(Pigeon_coeff_conversion([14.660,4.5322,1.8047,-2.9883,0,0]), 'Pigeon; 1996', "mm", None, 2.9326, 'Elbow'),\
								MA_Settings([36,0,0,0,0,0], 'Holzbaur; 2005', "mm", None, None, "Elbow")
							]}, \
		'Spindle' : Spindle_Settings(320,292.6,1.1,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006", \
					"Sources" : \
						[\
							Input_Source(163.8,"Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(11.6, 'Holzbaur; 2005', 'cm'),\
											Input_Source(14.22, "Garner & Pandy; 2003", "cm")
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(27.2, "Holzbaur; 2005", "cm"),\
											Input_Source(22.98, "Garner & Pandy; 2003", "cm")
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(0, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source((4.5+3.1), "Holzbaur; 2005", "sq cm"),\
							Input_Source(25.90, "Garner & Pandy; 2003", "sq cm")
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source((624.3+435.6), "Holzbaur; 2005", "N"),\
												Input_Source(849.29, "Garner & Pandy; 2003", "N")
											]}\
		}

	TRI_Settings = {\
		'Notes' : [\
					"TRI EFE MA for Ramsay; 2009 has R² = 0.997 whereas Pigeon has R² = 0.9904. Pigeon appears to really fail when the elbow angle is greater than 140°. For this reason, Ramsay; 2009 should be used. However the approach of fixing the MA for values greater than 140° can be adopted for completeness. Coefficients and equation number/type are listed below to test either implementation.",\
					"Note: Only using the long head for optimal length, see Holzbaur (2005) for additional head parameters.",\
					"Banks had the parameters for each head of the triceps, values were added.",\
					"Holzbaur settings only utilizes the long head of the TRI."\
					],
		'Shoulder MA' : {	"Primary Source" : "Pigeon; 1996",\
		 					"Sources" : \
								[\
									MA_Settings([-25.40,0,0,0,0,0], 'Pigeon; 1996', "mm", None, None, "Shoulder"),\
									MA_Settings(-15, 'Est', "mm", None, None, "Shoulder")\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Ramsay; 2009", \
	 					"Sources" : \
							[\
								MA_Settings([-24.5454,-8.8691,9.3509,-1.7518,0], 'Ramsay; 2009', 'mm', 1, None, 'Elbow'),\
								MA_Settings(Pigeon_coeff_conversion([-23.287,-3.0284,12.886,-19.092,13.277,-3.5171]), 'Pigeon; 1996', 'mm', None, None, 'Elbow'),\
								MA_Settings([-21,0,0,0,0,0], 'Holzbaur; 2005', 'mm', None, None, 'Elbow')
							]}, \
		'Spindle' : Spindle_Settings((200+222+98),(223.7+269.6+221.8),(0.89+0.82+0.44)/3,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006",\
					"Sources" : \
						[\
							Input_Source((94.2+138.4+92.5), "Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005",\
		 							"Sources" : \
										[\
											Input_Source(13.4, 'Holzbaur; 2005', 'cm'),\
											Input_Source(8.77, 'Garner & Pandy; 2003', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(14.3, "Holzbaur; 2005", 'cm'),\
											Input_Source(19.05, 'Garner & Pandy; 2003', 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(12, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source((5.7+4.5+4.5),"Holzbaur; 2005",'sq cm'),\
							Input_Source(76.30, 'Garner & Pandy; 2003', 'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source((798.5+624.3+624.3),"Holzbaur; 2005",'N'),\
												Input_Source(2332.92, 'Garner & Pandy; 2003', 'N')\
											]}\
		}

	BRA_Settings = {\
		"Notes" : [\
					"BRA (Brachialis) EFE MA for Ramsay; 2009 has R² = 0.990 whereas Pigeon has R² = 0.9988. Curve appears to be a better fit, as it experiences its smallest MA when Elbow angle = 0. Coefficients and equation number/type are listed below to test either implementation."\
					],
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Ramsay; 2009", \
	 					"Sources" : \
							[\
								MA_Settings([16.1991,-16.1463,24.5512,-6.3335,0], 'Ramsay; 2009', 'mm', 1, None, 'Elbow'),\
								MA_Settings(Pigeon_coeff_conversion([5.5492,2.3080,2.3425,-2.0530,0,0]), 'Pigeon; 1996', 'mm', None, None, 'Elbow'),\
								MA_Settings([18,0,0,0,0,0], 'Holzbaur; 2005', 'mm', None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(256,272.1,0.94,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006", \
					"Sources" : \
						[\
							Input_Source(141, "Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(8.6, 'Holzbaur; 2005', 'cm'),\
											Input_Source(10.28, 'Holzbaur; 2005', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(5.4, "Holzbaur; 2005", 'cm'),\
											Input_Source(1.75, "Holzbaur; 2005", 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(0, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(7.1,"Holzbaur; 2005",'sq cm'),\
							Input_Source(25.88,"Garner & Pandy; 2003",'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(987.3,"Holzbaur; 2005","N"),\
												Input_Source(583.76,"Garner & Pandy; 2003","N")\
											]}\
		}

	BRD_Settings = {\
		"Notes" : [\
					"BRD (Brachioradialis) for Ramsay; 2009 has R² = 0.988 whereas Pigeon has R² = 0.9989. Pigeon, however, only takes elbow angle into account, whereas Ramsay; 2009 takes in variable PS angles. Coefficients and equation number/type are listed below to test either implementation."\
					],
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Ramsay; 2009", \
	 					"Sources" : \
							[\
								MA_Settings(	[15.2564,-11.8355,2.8129,-5.7781,44.8143,0,2.9032,0,0,-13.4956,0,-0.3940,0,0,0,0], 'Ramsay; 2009', 'mm', 2, None, 'Elbow'), \
								MA_Settings(	Pigeon_coeff_conversion([19.490,1.6681,10.084,-6.5171,0,0]), 'Pigeon; 1996', 'mm', None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(70,190.2,0.37,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006",\
					"Sources" : \
						[\
							Input_Source(64.7,"Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(17.3, 'Holzbaur; 2005', 'cm'),\
											Input_Source(27.03, 'Garner & Pandy; 2003', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(13.3, "Holzbaur; 2005", 'cm'),\
											Input_Source(6.04, "Garner & Pandy; 2003", 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(0, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(1.9,"Holzbaur; 2005",'sq cm'),\
							Input_Source(3.08,"Garner & Pandy; 2003",'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(261.3,"Holzbaur; 2005",'N'),\
												Input_Source(101.56,"Garner & Pandy; 2003",'N')\
											]}\
		}

	PRO_Settings = {\
		'Notes' : [\
					""\
					],
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Ramsay; 2009", \
	 					"Sources" : \
							[\
								MA_Settings(	[11.0405,-1.0079,0.3933,-10.4824,-12.1639,-0.4369,36.9174,3.5232,-10.4223,21.2604,-37.2444,10.2666,-11.0060,14.5974,-3.9919,1.7526,-2.0089,0.5460], 'Ramsay; 2009', 'mm', 3, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(187.6,185.5,1.3,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006",\
					"Sources" : \
						[\
							Input_Source(38.8, "Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(4.9, 'Holzbaur; 2005', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(9.8, "Holzbaur; 2005", 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(10, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(4.0,"Holzbaur; 2005",'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(566.2,"Holzbaur; 2005",'N')\
											]}\
		}

	FCR_Settings = {\
		'Notes' : [\
					"FCR EFE MA is not listed in Ramsay; 2009 but Pigeon has a quadratic function with R² = 0.9975. Pigeon only takes elbow angle into account. Coefficients and equation number/type are listed below to test either implementation. EFE MA was estimated to be constant and 10 mm for this muscle. If you use Pigeon, make sure to only accept positive moment arm values, as this model fails outside the ROM. One option is to set the MA to the constant value (i.e., MA[theta>140°] = MA[140°])"\
					],\
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Pigeon; 1996", \
	 					"Sources" : \
							[\
								MA_Settings(Pigeon_coeff_conversion([0.9351,0.5375,-0.3627,0,0,0]), 'Pigeon; 1996', 'mm', None, 2.86, 'Elbow'),\
								MA_Settings(1, 'Est', 'cm', None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(129,125.7,1.0,"Banks; 2006"),\
		'Mass' : (28.7, "Banks; 2006", 'g'),\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(6.3, 'Holzbaur; 2005', 'cm'),\
											Input_Source(5.10, 'Garner & Pandy; 2003', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(24.4, "Holzbaur; 2005", 'cm'),\
											Input_Source(27.08, "Garner & Pandy; 2003", 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(3, "Holzbaur; 2005", 'deg')
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(1.6,"Holzbaur; 2005",'sq cm'),\
							Input_Source(11.16,"Garner & Pandy; 2003",'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(74.0,"Holzbaur; 2005","N"),\
												Input_Source(368.63,"Garner & Pandy; 2003","N")\
											]}\
		}

	ECRB_Settings = {\
		'Notes' : [\
					"Garner and Pandy do not distinguish between ERCB and ECRL. Parameter values were included for completeness."\
					],\
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Ramsay; 2009", \
	 					"Sources" : \
							[\
								MA_Settings([-11.256,17.8548,1.6398,-0.5073,-2.8827,0,-0.0942,0,0,0,0,0,0,0,0,0], 'Ramsay; 2009', 'mm', 2, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(102,132.7,0.77,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006",\
					"Sources" : \
						[\
							Input_Source(32.1, "Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(5.9, 'Holzbaur; 2005', 'cm'),\
											Input_Source(7.28, 'Garner & Pandy; 2003', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(22.2, "Holzbaur; 2005", 'cm'),\
											Input_Source(26.80, "Garner & Pandy; 2003", 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(9, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(2.2,"Holzbaur; 2005",'sq cm'),\
							Input_Source(24.89,"Garner & Pandy; 2003",'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(100.5,"Holzbaur; 2005","N"),\
												Input_Source(755.76,"Garner & Pandy; 2003","N")\
											]}\
		}

	ECRL_Settings = {\
		'Notes' : [\
					"ECRL EFE MA for Ramsay; 2009 has R² = 0.978 whereas Pigeon has R² = 0.9986. Pigeon, however, only takes elbow angle into account, whereas Ramsay; 2009 takes in variable PS angles. Additionally, Pigeon only considers elbow angles between 0 and 140 degrees and exhibits a decrease in MA as elbow angle approaches the upper bound of the ROM. This should (intiutively speaking) make the extensors MA largest, but Pigeon exhibits a drop off that may make it less favorable for movements at the boundary of the ROM. Coefficients and equation number/type are listed below to test either implementation.",\
					"Garner and Pandy do not distinguish between ERCB and ECRL. Parameter values were included for completeness. Might need to divide by 2 to 'split' muscle."\
					],
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Pigeon; 1996", \
	 					"Sources" : \
							[\
								MA_Settings(Pigeon_coeff_conversion([4.7304,1.2590,4.4347,-3.0229,0,0]), 'Pigeon; 1996', 'mm', None, None, 'Elbow'),\
								MA_Settings([-7.7034,16.3913,7.4361,-1.7566,0,-1.3336,0,0.0742,0,0,0,0,0,0,0,0], 'Ramsay; 2009', 'mm', 2, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(74,155.2,0.48,"Banks; 2006"),\
		'Mass' : { "Primary Source" : "Banks; 2006", \
					"Sources" : \
						[\
							Input_Source(44.3, "Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(8.1, 'Holzbaur; 2005', 'cm'),\
											Input_Source(7.28, 'Garner & Pandy; 2003', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(22.4, "Holzbaur; 2005", 'cm'),\
											Input_Source(26.80, "Garner & Pandy; 2003", 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(0, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(2.2,"Holzbaur; 2005",'sq cm'),\
							Input_Source(24.89,"Garner & Pandy; 2003",'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(304.9,"Holzbaur; 2005",'N'),\
												Input_Source(755.76,"Garner & Pandy; 2003",'N')\
											]}\
		}

	FCU_Settings = {\
		'Notes' : [\
					""\
					],
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Est", \
	 					"Sources" : \
							[\
								MA_Settings(1, 'Est', 'cm', None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(175,141.2,1.2,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006", \
					"Sources" : \
						[\
							Input_Source(36.5,"Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(5.1, 'Holzbaur; 2005', 'cm'),\
											Input_Source(3.98, 'Garner & Pandy; 2003', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(26.5, "Holzbaur; 2005", 'cm'),\
											Input_Source(27.14, "Garner & Pandy; 2003", 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(12, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(2.9,"Holzbaur; 2005",'sq cm'),\
							Input_Source(16.99,"Garner & Pandy; 2003",'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(128.9,"Holzbaur; 2005",'N'),\
												Input_Source(561.00,"Garner & Pandy; 2003",'N')\
											]}\
		}

	FDS_Settings = {\
		'Notes' : [\
					"Note: only they muscle for the second digit was used for the FDS muscle. NEED TO DETERMINE IF THIS SHOULD BE A SUM OR AN AVERAGE FOR MEASURES LIKE PCSA, F_MAX, ETC.",\
					"As we are only considering one digit, do we need to sum all of the peak forces in order to get a better representation of its force producing capabilities?"\
					],
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Est", \
	 					"Sources" : \
							[\
								MA_Settings(1, 'Est', 'cm', None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(356,224.9,1.6,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006",\
					"Sources" : \
						[\
							Input_Source(95.2,"Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(8.4, 'Holzbaur; 2005', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(27.5, "Holzbaur; 2005", 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(6, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(1.4,"Holzbaur; 2005",'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(61.2,"Holzbaur; 2005",'N')\
											]}\
		}

	PL_Settings = {\
		'Notes' : [\
					""\
					],
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Est", \
	 					"Sources" : \
							[\
								MA_Settings(1, 'Est', 'cm', None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(None,None,None,None),\
		'Mass' : {	"Primary Source" : None,\
					"Sources" : \
						[\
							Input_Source(None, "N/A","N/A")\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(6.4, 'Holzbaur; 2005', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(26.9, "Holzbaur; 2005", 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(4, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(0.6,"Holzbaur; 2005",'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(26.7,"Holzbaur; 2005",'N')\
											]}\
		}

	ECU_Settings = {\
		'Notes' : [\
					"ECU EFE MA is not listed in Ramsay; 2009 but Pigeon has a quadratic function with R² = 0.9966. Pigeon only takes elbow angle into account. Coefficients and equation number/type are listed below to test either implementation. EFE MA was estimated to be constant and -10 mm for this muscle. If you use Pigeon, make sure to only accept negative moment arm values, as this model fails outside the ROM. One option is to set the MA to the constant value (i.e., MA[theta>140°] = MA[140°])"\
					],
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Pigeon; 1996", \
	 					"Sources" : \
							[\
								MA_Settings(Pigeon_coeff_conversion([-2.1826,-1.7386,1.1491,0,0,0]), 'Pigeon; 1996', 'mm', None, None, 'Elbow'),\
								MA_Settings(-1, 'Est', 'cm', None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(157,118,1.3,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006",\
					"Sources" : \
						[\
							Input_Source(25.2,"Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(6.2, 'Holzbaur; 2005', 'cm'),\
											Input_Source(3.56, 'Garner & Pandy; 2003', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(26.5, "Holzbaur; 2005", 'cm'),\
											Input_Source(28.18, "Garner & Pandy; 2003", 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(12, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(2.9,"Holzbaur; 2005",'sq cm'),\
							Input_Source(8.04,"Garner & Pandy; 2003",'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(128.9,"Holzbaur; 2005",'N'),\
												Input_Source(265.58,"Garner & Pandy; 2003",'N')\
											]}\
		}

	EDM_Settings = {\
		'Notes' : [\
					""\
					],
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Est", \
	 					"Sources" : \
							[\
								MA_Settings(-1, 'Est', 'cm', None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(53,59.8,0.89,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006", \
					"Sources" : \
						[\
							Input_Source(6.2, "Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(6.8, 'Holzbaur; 2005', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : 'Holzbaur; 2005', \
									"Sources" : \
										[\
											Input_Source(32.2, 'Holzbaur; 2005', 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : 'Holzbaur; 2005', \
								"Sources" : \
									[\
										Input_Source(3, 'Holzbaur; 2005', 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : 'Holzbaur; 2005', \
					'Sources' : \
						[\
							Input_Source(0.6,'Holzbaur; 2005','sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : 'Holzbaur; 2005', \
										'Sources': \
											[\
												Input_Source(25.3,'Holzbaur; 2005','N')\
											]}\
		}

	EDC_Settings = {\
		'Notes' : [\
					"Note: only they muscle for the second digit was used for the EDC muscle. NEED TO DETERMINE IF THIS SHOULD BE A SUM OR AN AVERAGE FOR MEASURES LIKE PCSA, F_MAX, ETC."\
					],
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Est", \
	 					"Sources" : \
							[\
								MA_Settings(-1, 'Est', 'cm', None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(219,152.6,1.4,"Banks; 2006"),\
		'Mass' : {	"Primary Source" : "Banks; 2006", \
					"Sources" : \
						[\
							Input_Source(42.8, "Banks; 2006", 'g')\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(7.0, 'Holzbaur; 2005', 'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(32.2, "Holzbaur; 2005", 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(3, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(18.3,"Holzbaur; 2005",'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(0.4,"Holzbaur; 2005",'N')\
											]}\
		}

	AN_Settings = {\
		'Notes' : [\
					""\
					],
		'Shoulder MA' : {	"Primary Source" : "Est",\
		 					"Sources" : \
								[\
									MA_Settings(0, 'Est', 'm', None, None, 'Shoulder')\
								]}, \
		'Elbow MA' : {	"Primary Source" : "Pigeon; 1996", \
	 					"Sources" : \
							[\
								MA_Settings(Pigeon_coeff_conversion([-5.3450,-2.2841,8.4297,-14.329,10.448,-2.736]), 'Pigeon; 1996', 'mm', None, None, 'Elbow')\
							]}, \
		'Spindle' : Spindle_Settings(None,None,None,None),\
		'Mass' : {	"Primary Source" : None, \
					"Sources" : \
						[\
							Input_Source(None, None, None)\
						]},\
		'Optimal Muscle Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(2.7,"Holzbaur; 2005",'cm')\
										]},\
		'Optimal Tendon Length' : {	"Primary Source" : "Holzbaur; 2005", \
									"Sources" : \
										[\
											Input_Source(1.8, "Holzbaur; 2005", 'cm')\
										]}, \
		'Pennation Angle' : {	"Primary Source" : "Holzbaur; 2005", \
								"Sources" : \
									[\
										Input_Source(0, "Holzbaur; 2005", 'deg')\
									]}, \
		'PCSA' : {	"Primary Source" : "Holzbaur; 2005", \
					'Sources' : \
						[\
							Input_Source(2.5,"Holzbaur; 2005",'sq cm')\
						]}, \
		'Maximum Isometric Force': {	"Primary Source" : "Holzbaur; 2005", \
										'Sources': \
											[\
												Input_Source(350.0,"Holzbaur; 2005",'N')\
											]}\
		}

	AllAvailableMuscles =[	"PC", "DELTa", "CB", "DELTp", "BIC", \
							"TRI", "BRA", "BRD", "PRO", "FCR",\
	 						"ECRB", "ECRL", "FCU", "FDS", "PL",\
	  						"ECU", "EDM", "EDC", "AN"]
	AllMuscleSettings = {	'PC': PC_Settings, 'DELTa' : DELTa_Settings, \
							'CB' : CB_Settings, 'DELTp' : DELTp_Settings,\
							'BIC' : BIC_Settings, 'TRI' : TRI_Settings, \
							'BRA' : BRA_Settings, 'BRD' : BRD_Settings, \
							'PRO' : PRO_Settings, 'FCR' : FCR_Settings, \
							'ECRB' : ECRB_Settings, 'ECRL' : ECRL_Settings, \
							'FCU' : FCU_Settings, 'FDS' : FDS_Settings, \
							'PL' : PL_Settings,'ECU' : ECU_Settings, \
							'EDM' : EDM_Settings, 'EDC' : EDC_Settings,\
							'AN' : AN_Settings}
	if PreselectedMuscles is None:
		ValidResponse_1 = False
		while ValidResponse_1 == False:
			MuscleSelectionType = input("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nMuscle Selection:\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n (1) - Default\n (2) - Custom\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nResponse: ")
			# import ipdb; ipdb.set_trace()
			print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
			if MuscleSelectionType not in ['1','2','']:
				print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nInvalid Response! Please try again.')
				ValidResponse_1 = False
			elif MuscleSelectionType == '' or MuscleSelectionType == '1':
				for Muscle in ["PRO","AN"]:
					del(AllMuscleSettings[Muscle])
				ValidResponse_1 = True
			elif MuscleSelectionType == '2':
				ValidResponse_2 = False
				while ValidResponse_2 == False:
					MuscleListString = ""
					for i in range(len(AllAvailableMuscles)):
						MuscleListString += " (" + str(i+1) + ") - " + AllAvailableMuscles[i] + "\n"
					MuscleSelectionNumbers = input("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nSelect Muscle Number(s)\n(separated by commas & groups with hyphens):\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n" + MuscleListString + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nMuscle Number(s): ")
					print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
					MuscleSelectionNumbers = [el.strip() for el in MuscleSelectionNumbers.split(",")]
					for el in MuscleSelectionNumbers:
						if "-" in el:
							temp = el.split("-")
							MuscleSelectionNumbers.remove(el)
							[MuscleSelectionNumbers.append(str(i)) \
											for i in range(int(temp[0]),int(temp[1])+1)]
					if np.array([el in [str(i+1) for i in range(len(AllAvailableMuscles))] \
										for el in MuscleSelectionNumbers]).all() == False:
						print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nInvalid Response! Please check muscle numbers and try again.')
						ValidResponse_2 = False
					else:
						SelectedMuscles = [AllAvailableMuscles[int(el)-1] \
												for el in MuscleSelectionNumbers]
						MusclesToBeDeleted = [Muscle for Muscle in AllAvailableMuscles \
													if Muscle not in SelectedMuscles]
						for Muscle in MusclesToBeDeleted:
							del(AllMuscleSettings[Muscle])
						ValidResponse_2 = True
				ValidResponse_1 = True
	else:
		# assert type(PreselectedMuscles)==list and len(PreselectedMuscles)==8, "PreselectedMuscles, when used, must be a list of 8 numbers."
		assert np.array([type(MuscleNumber)==int for MuscleNumber in PreselectedMuscles]).all(),\
			"PreselectedMuscles must be a list of muscle numbers (ints)."
		assert np.array([MuscleNumber in range(1,len(AllAvailableMuscles)+1) \
			for MuscleNumber in PreselectedMuscles]).all(), \
				"PreselectedMuscles contains a muscle number outside the available muscles."
		SelectedMuscles = [AllAvailableMuscles[int(el)-1] \
								for el in PreselectedMuscles]
		MusclesToBeDeleted = [Muscle for Muscle in AllAvailableMuscles \
									if Muscle not in SelectedMuscles]
		for Muscle in MusclesToBeDeleted:
			del(AllMuscleSettings[Muscle])
	return(AllMuscleSettings)

def return_primary_source(Settings):
	assert Settings["Primary Source"]!=None, "No sources were found for this setting."
	TotalSources = Settings["Sources"]
	PrimarySource = Settings["Primary Source"]
	assert PrimarySource in [settings.Source for settings in TotalSources], "Error! Primary Source is not referenced."
	return(TotalSources[np.where([settings.Source == PrimarySource for settings in TotalSources])[0][0]])

def unit_conversion(Params):
    assert hasattr(Params,"Units"), "Params must have attr 'Units' in order for unit_conversion to work."
    assert hasattr(Params,"Values"), "Params must have attr 'Values' in order for unit_conversion to work."

    Units = Params.Units
    Value = Params.Values
    assert Units.capitalize() in \
            ["Degrees","Deg","Degree",
            "Radians","Radian","Rad",
            "Rads","In","Inches",
            "Cm","Centimeters","Centimeter",
            "Mm","Millimeters","Millimeter",
            "Meters","Meter","M",
            "Sq in","Squared inches","Inches squared",
            "In sq","Cm sq","Sq cm",
            "Centimeters squared","Squared centimeters","Mm sq",
            "Sq mm","Millimeters squared","Squared millimeters",
            "Meters squared","Squared meter","M sq",
            "Sq m","Lbs","Lb",
            "Pounds","G","Grams",
            "Gram","Kg","Kilograms",
            "Kilogram","N","Newton",
            "Newtons"], \
        "Improper Units Value. Please use appropriate Units."

    if Units.capitalize() in \
            ["Degrees","Deg","Degree",
            "Radians","Radian","Rad",
            "Rads"]:
    	if Units.capitalize() in ["Radians","Radian","Rad","Rads"]:
    		return(Value)
    	elif Units.capitalize() in ["Degrees","Deg","Degree"]:
    		if type(Value)==list:
    			return(list(np.array(Value)*np.pi/180))
    		else:
    			return(Value*np.pi/180)

    elif Units.capitalize() in \
            ["In","Inches","Cm",
            "Centimeters","Centimeter","Mm",
    		"Millimeters","Millimeter","Meters",
            "Meter","M"]:
    	if Units.capitalize() in ["Meter","Meters","M"]:
    		return(Value)
    	elif Units.capitalize() in ["In","Inches"]:
    		if type(Value)==list:
    			return(list(np.array(Value)*2.54/100))
    		else:
    			return(Value*2.54/100)
    	elif Units.capitalize() in  ["Cm","Centimeters","Centimeter"]:
    		if type(Value)==list:
    			return(list(np.array(Value)/100))
    		else:
    			return(Value/100)
    	elif Units.capitalize() in  ["Mm","Millimeters","Millimeter"]:
    		if type(Value)==list:
    			return(list(np.array(Value)/1000))
    		else:
    			return(Value/1000)

    elif Units.capitalize() in \
            ["Sq in","Squared inches","Inches squared",
            "In sq","Cm sq","Sq cm",
    		"Centimeters squared","Squared centimeters","Mm sq",
            "Sq mm","Millimeters squared","Squared millimeters",
            "Meters squared","Squared meter","M sq",
            "Sq m"]:
    	if Units.capitalize() in ["Meters squared","Squared meter","M sq","Sq m"]:
    		return(Value)
    	elif Units.capitalize() in ["Sq in","Squared inches","Inches squared","In sq"]:
    		if type(Value)==list:
    			return(list(np.array(Value)*((2.54/100)**2)))
    		else:
    			return(Value*((2.54/100)**2))
    	elif Units.capitalize() in ["Cm sq","Sq cm","Centimeters squared","Squared centimeters"]:
    		if type(Value)==list:
    			return(list(np.array(Value)/(100**2)))
    		else:
    			return(Value/(100**2))
    	elif Units.capitalize() in ["Mm sq","Sq mm","Millimeters squared","Squared millimeters"]:
    		if type(Value)==list:
    			return(list(np.array(Value)/(1000**2)))
    		else:
    			return(Value/(1000**2))

    elif Units.capitalize() in ["Lbs","Lb","Pounds","G","Grams","Gram","Kg","Kilograms","Kilogram"]:
    	if Units.capitalize() in ["Kg","Kilograms","Kilogram"]:
    		return(Value)
    	elif Units.capitalize() in ["G","Grams","Gram"]:
    		if type(Value)==list:
    			return(list(np.array(Value)/1000))
    		else:
    			return(Value/1000)
    	elif Units.capitalize() in  ["Lbs","Lb","Pounds"]:
    		if type(Value)==list:
    			return(list(np.array(Value)*0.45359237))
    		else:
    			return(Value*0.45359237)

    elif Units.capitalize() in ["N","Newton","Newtons"]:
    	if Units.capitalize() in ["N","Newton","Newtons"]:
    		return(Value)
