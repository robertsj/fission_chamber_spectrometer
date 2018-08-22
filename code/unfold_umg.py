from unfolding_tools import Unfolding
import pickle
import scipy as sp
import numpy as np
from unfold import integral_response
from flux_spectrum import Flux
from master_data import isos, data_path, plot_path
from response import generate_responses
from multigroup_utilities import energy_groups
from spectrum import Spectrum
import matplotlib.pyplot as plt
import os
import shutil
from scipy.integrate import trapz
from flux import select_flux_spectrum


def unfold_umg(R, RF, ds, ts, eb, isos, name, program):
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
    U.set_routine(program)
    U.run(name)
    # plotit(U, ts, name)
    shutil.rmtree('inp')
    return U.solution, 100 * abs(U.solution.values - ts.values) / ts.values


def plotit(U, ts, name):
    # load dataset
    sol = U.solution

    fig = plt.figure(123)
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('E (MeV)')
    ax.set_ylabel('$\Phi$')

    ax.plot(*U.ds.step, color='k', label='Default')
    ax.plot(*ts.step, color='r', linestyle='--', label='True')
    ax.plot(*sol.step, color='b', linestyle=':', label='MAXED')

    ax.legend(frameon=False)
    fig.savefig(plot_path + name + '.pdf')
    plt.close(fig)

    # plot comparisons
    fig = plt.figure(124)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Group')
    ax.set_ylabel('$\Phi_{MAXED}/$\Phi_{true}$')

    x = range(U.ds.num_bins)
    y = 100 * abs(sol.values - ts.values) / ts.values
    y = y[::-1]
    ax.plot(x, y, color='k', marker='o')
    fig.savefig(plot_path + name + '_comp.pdf')
    plt.close(fig)

    os.system('rm ' + data_path + '{}_unfolded.txt'.format(name))


def bin_flux(flux, struct):
    lower = 1e-5
    upper = 2e7
    eb = energy_groups(struct, lower, upper)
    phi = sp.zeros(len(eb)-1)
    for i in range(len(eb)-1):
        E = sp.logspace(sp.log10(eb[i+1]), sp.log10(eb[i]), 1e4)
        x = trapz(flux(E), E)
        print("->", i, eb[i+1], eb[i], x)
        phi[i] = trapz(flux(E), E)
    phi = phi[::-1] / np.sum(phi)
    return phi


isos_cd = ['u233cd113', 'u235cd113', 'pu238cd113', 'pu239cd113', 'pu241cd113']
isos_gd = ['u233gd155', 'u235gd155', 'pu238gd155', 'pu239gd155', 'pu241gd155']

isos_1 = ['u235', 'u238', 'th232']
isos_2 = ['u235', 'u238', 'th232', 'np237', 'pu238']
isos_3 = ['th232', 'u233', 'u234', 'u235', 'u238', 'np237',
          'pu238', 'pu239', 'pu240', 'pu241', 'pu242']
isos_4 = isos
isos_5 = isos_3 + isos_cd
isos_6 = isos_3 + isos_gd
isos_7 = isos_3 + isos_cd + isos_gd

structs = ['tg0_625', 'lwr32', 'wims69']
umg_data = {}
for struct in structs:
    phi_triga = select_flux_spectrum('trigaC', 1)[2]
    name = 'test_wims69_resp.p'
    resp = generate_responses(isos+isos_cd+isos_gd, phi_triga,
                              struct=struct, name=name, overwrite=True)
    R = integral_response(name)
    RF = resp['response']
    for key, val in RF.items():
        RF[key] = val[::-1]

    phi_ref = resp['phi'][::-1]
    phi_ref = phi_ref / sum(phi_ref)
    eb = resp['eb'][::-1] * 1e-6  # convert to MeV
    ts = Spectrum(eb, phi_ref)

    pwr = Flux(7.0, 600.0)
    ds_pwr = bin_flux(pwr.evaluate, struct)
    ds_low = Spectrum(eb, ds_pwr * 0.5)
    ds_hi = Spectrum(eb, ds_pwr * 2)
    ds_on = Spectrum(eb, ds_pwr)
    ds_ones = Spectrum(eb, ds_pwr * 0 + 1)
    default_spectra = {'hi': ds_hi,
                       'low': ds_low,
                       'on': ds_on,
                       'ones': ds_ones}

    all_iso_sets = [isos_1, isos_2, isos_3, isos_4, isos_5, isos_6, isos_7]
    for i, iso_set in enumerate(all_iso_sets):
        for key, ds in default_spectra.items():
            for p, pgm in [('mx', 'maxed'), ('gr', 'gravel')]:
                nombre = '{}_{}_{}_iso{}'.format(struct, key, p, i+1)
                umg_data[nombre] = unfold_umg(R, RF, ds, ts, eb, isos=iso_set, name=nombre, program=pgm)

with open(data_path + 'umg_data.p', 'wb') as F:
    pickle.dump(umg_data, F)
