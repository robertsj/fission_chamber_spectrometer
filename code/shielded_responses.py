# use cd-113, gd-155, or au-197 to shield a fission chamber response

import matplotlib.pyplot as plt
import numpy as np
from master_data import isos_str, directory, img_directory

from nice_plots import init_nice_plots
init_nice_plots()

isos_colors = {'u233': 'b', 'u235': 'r', 'pu238': 'g', 'pu239': 'c', 'pu241': 'm'}

# Limit to the simpler thermal sensitives
isos_th = ['u233','u235',
           'pu238', 'pu239','pu241']


filt = {}

filt['cd113'] = {}
filt['cd113']['numd'] = 6.022e23 / 113.0 * 8.65 / 1e24 * 0.12
filt['cd113']['thick'] = 0.025

filt['gd155'] = {}
filt['gd155']['numd'] = 6.022e23 / 155.0 * 7.9 / 1e24 * 0.148
filt['gd155']['thick'] = 0.025

filt['au197'] = {}
filt['au197']['numd'] = 6.022e23 / 197.0 * 19.3 / 1e24 * 0.148
filt['au197']['thick'] = 0.05

plt.figure(1)
for iso in ['cd113', 'gd155', 'au197'] :
    # Load in the filter data
     E, sig = np.loadtxt(directory+'/data/'+iso+'.txt', skiprows=1, 
                            delimiter=',', unpack=True) 
     filt[iso]['value'] = sig
     filt[iso]['E'] = E
     plt.loglog(E, sig)
plt.xlabel('$E$ (eV)')
plt.ylabel('$\sigma_{\gamma}(E)$ (b)')
#plt.axis([1e-5, 1e7, 1e-6, 1e1])
plt.savefig(img_directory+'thermal_filters.pdf')

plt.figure(2)
for iso in isos_th :
    # Load in the filter data
    E, sig = np.loadtxt(directory+'/data/'+iso+'.txt', skiprows=1, 
                        delimiter=',', unpack=True) 
    EE = np.logspace(-5, np.log10(2e7), int(1e4))
    isof = 'cd113'
    coef = -filt[isof]['numd']*filt[isof]['thick']
    sig_filt = np.interp(EE,E, sig) * \
                   np.exp(coef*np.interp(EE, filt[isof]['E'], filt[isof]['value']))
    plt.loglog(E, sig, label=isos_str[iso], color=isos_colors[iso],alpha=0.5)          
    plt.loglog(EE, sig_filt, label=isos_str[iso]+isos_str[isof], color=isos_colors[iso],ls='--')
    np.savetxt(directory+'/data/'+iso+isof+'.txt', np.array([EE, sig_filt]).T, delimiter=',', header=iso+isof)
plt.xlabel('$E$ (eV)')
plt.ylabel('$\sigma_{\gamma}(E)$ (b)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.axis([1e-5, 1e7, 1e-6, 1e6])
plt.savefig(img_directory+'filtered_responses_cd.pdf', bbox_inches='tight')

plt.figure(3)
for iso in isos_th :
    # Load in the filter data
    E, sig = np.loadtxt(directory+'/data/'+iso+'.txt', skiprows=1, 
                        delimiter=',', unpack=True) 
    EE = np.logspace(-5, np.log10(2e7), int(1e4))
    isof = 'gd155'
    coef = -filt[isof]['numd']*filt[isof]['thick']
    sig_filt = np.interp(EE,E, sig) * \
                   np.exp(coef*np.interp(EE, filt[isof]['E'], filt[isof]['value']))
    plt.loglog(E, sig, label=isos_str[iso], color=isos_colors[iso],alpha=0.5)          
    plt.loglog(EE, sig_filt, label=isos_str[iso]+isos_str[isof], color=isos_colors[iso],ls='--')
    np.savetxt(directory+'/data/'+iso+isof+'.txt', np.array([EE, sig_filt]).T, delimiter=',', header=iso+isof)
plt.xlabel('$E$ (eV)')
plt.ylabel('$\sigma_{\gamma}(E)$ (b)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.axis([1e-5, 1e7, 1e-6, 1e6])
plt.savefig(img_directory+'filtered_responses_gd.pdf', bbox_inches='tight')
