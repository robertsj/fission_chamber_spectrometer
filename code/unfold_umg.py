from unfolding_tools import Unfolding
import pickle
import scipy as sp
import numpy as np
from scipy.optimize import fmin_cobyla, minimize
from master_data import directory
from unfold import integral_response
from flux_spectrum import Flux
from master_data import isos, img_directory, data_path, plot_path
from response import generate_responses
from multigroup_utilities import energy_groups, plot_multigroup_data
from spectrum import Spectrum
import matplotlib.pyplot as plt
import shutil




def unfold_umg(R, RF, ds, rs, eb, isos, name):
    # generate response vector
    response = np.ones(len(isos))
    rfs = np.ones((len(isos), len(ds.values)))
    ids = []
    for i, iso in enumerate(isos):
        response[i] = R[iso]
        rfs[i] = RF[iso]
        names = ['{:8.8}'.format(iso), '{:16.16}'.format(iso)]
        ids.append(names)
        
    U = Unfolding()
    U.setSphereIDs(ids)
    U.setSphereSizes(np.ones(len(isos)))
    U.set_responses(response)
    U.set_rf(eb, rfs)
    U.set_ds(ds)
    U.run(name)
    plotit(U, rs, name)
    shutil.rmtree('inp')


def plotit(U, rs, name):
    # load dataset
    data = np.loadtxt(data_path + '{}_unfolded.txt'.format(name))
    data = data.T
    sol = Spectrum(U.ds.edges, data[1], data[2], dfde=True)
        
    fig = plt.figure(123)
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.plot(*U.ds.step, color='k', label='Default')
    ax.plot(*rs.step, color='r', linestyle='--', label='Starting')
    ax.plot(*sol.step, color='b', label='Solution')
    
    ax.legend(frameon=False)
    fig.savefig(plot_path + name + '.pdf')


isos_cd = ['u233cd113', 'u235cd113', 'pu238cd113', 'pu239cd113', 'pu241cd113']
isos_gd = ['u233gd155', 'u235gd155', 'pu238gd155', 'pu239gd155', 'pu241gd155']
  

struct = 'wims69'
pwr = Flux(7.0, 600.0)
name = 'test_wims69_resp.p'
resp = generate_responses(isos+isos_cd+isos_gd, pwr.evaluate, 
                          struct=struct, name=name, overwrite=True)
R = integral_response(name)
RF = resp['response']
for key, val in RF.items():
    RF[key] = val[::-1]

phi_ref = resp['phi'][::-1]
phi_ref = phi_ref / sum(phi_ref)
eb = resp['eb'][::-1] * 1e-6  # convert to MeV
ds = Spectrum(eb, phi_ref * 0.3)
rs = Spectrum(eb, phi_ref)

isos_1 = ['u235', 'u238', 'th232']
isos_2 = ['u235', 'u238', 'th232', 'np237', 'pu238']
isos_3 = ['th232','u233','u234','u235','u238','np237',
          'pu238','pu239','pu240','pu241','pu242']
isos_4 = isos


isos_5 = isos_3 + isos_cd
isos_6 = isos_3 + isos_gd
isos_7 = isos_3 + isos_cd + isos_gd

phi_3 = unfold_umg(R, RF, ds, rs, eb, isos=isos_1, name='generic')