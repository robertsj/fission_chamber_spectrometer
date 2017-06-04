isos = ['th232',
        'u233','u234','u235','u238',
        'np237',
        'pu238', 'pu239','pu240','pu241','pu242',
        'am241', 'am242m',
        'cm244', 'cm245']
       
       
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
            'cm245': '${}^{245}$Cm'}

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