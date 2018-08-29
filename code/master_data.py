directory = '/home/user/workspace/fission_chamber_spectrometer'
img_directory = directory + '/img/'

code_path = directory + '/code/'
plot_path = directory + '/plot/'
data_path = directory + '/data/'

umg_path = '/home/user/opt/U_M_G/FC/bin/'

isos = ['th232',
        'u233','u234','u235','u238',
        'np237',
        'pu238', 'pu239','pu240','pu241','pu242',
        'am241', 'am242m',
        'cm244', 'cm245']
       
isos_th = ['u233','u235',
           'pu238', 'pu239','pu241']

isos_gd  = ['u233gd155','u235gd155',
           'pu238gd155', 'pu239gd155','pu241gd155']
     
isos_str = {'th232': '${}^{232}$Th',
            'u233': '${}^{233}$U',
            'u234': '${}^{234}$U',
            'u235': '${}^{235}$U',
            'u238': '${}^{238}$U',
            'np237': '${}^{237}$Np',
            'pu238': '${}^{238}$Pu', 
            'pu239': '${}^{239}$Pu', 
            'pu240': '${}^{240}$Pu',
            'pu241': '${}^{241}$Pu', 
            'pu242': '${}^{242}$Pu',
            'am241': '${}^{241}$Am', 
            'am242m': '${}^{242m}$Am', 
            'cm244': '${}^{244}$Cm', 
            'cm245': '${}^{245}$Cm',
            'b10': '${}^{10}$B',
            'li6': '${}^{6}$Li',
            'ni59': '${}^{59}$Ni',
            'cd113': '${}^{113}$Cd',
            'gd155': '${}^{135}$Gd',
            'au197': '${}^{197}$Au',
            'u233cd113': '${}^{233}$U${}^{113}$Gd',
            'u233gd155': '${}^{233}$U${}^{155}$Gd',
            'u235cd113': '${}^{235}$U${}^{113}$Gd',
            'u235gd155': '${}^{235}$U${}^{155}$Gd',
            'pu238cd113': '${}^{238}$Pu${}^{113}$Gd',
            'pu238gd155': '${}^{238}$Pu${}^{155}$Gd',
            'pu239cd113': '${}^{239}$Pu${}^{113}$Gd',
            'pu239gd155': '${}^{239}$Pu${}^{155}$Gd',
            'pu241cd113': '${}^{241}$Pu${}^{113}$Gd',
            'pu241gd155': '${}^{241}$Pu${}^{155}$Gd'        
}


isos_colors = {}
import numpy as np
import matplotlib.pyplot as plt
colors = plt.get_cmap('Paired')(np.linspace(0, 1.0, len(isos)))
for i in range(len(colors)) :
    isos_colors[isos[i]] = colors[i]


"""
plan of attack
+ get cross sections loaded
+ get interpolating functions for the list
+ plot to verify (and use good color scheme)
- set up master script that will 
  - condense to a given structure assuming
    a given weighting spectrum
  - condense a given spectrum into a given 
    structure
  - save all the resulting information in a 
    binary pickle 
  - this represents the needed "response" 
    information
- set up script that uses interpolants
  to generate sensitivities (i.e., energy-
  integrated response)
- set up another script that 
  - loads in the responses
  - condenses a "default" spectrum to the 
    same group structure
  - unfolds the responses using a list of isos, 
    possibly incomplete
  - save the output 
- a post-process file that will load all the 
  cases of interest
"""
