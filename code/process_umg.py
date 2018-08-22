import scipy as sp
import matplotlib.pyplot as plt
import pickle
from master_data import isos, data_path, plot_path


with open(data_path + 'umg_data.p', 'rb') as F:
    umg_data = pickle.load(F)


def make_ds_plot(umg_data, pgm):
    # separate data
    ts = umg_data['wims69_ones_gr_iso4'][2]
    ds = {}
    ds_sol = {}
    ds_sol_err = {}
    for lab, spec in [('Unitary', 'wims69_ones_{}_iso4'.format(pgm)),
                      ('Same $\Phi$', 'wims69_on_{}_iso4'.format(pgm)),
                      ('High $\Phi$', 'wims69_hi_{}_iso4'.format(pgm)),
                      ('Low $\Phi$', 'wims69_low_{}_iso4'.format(pgm))]:
        ds[lab] = umg_data[spec][0].ds
        ds_sol[lab] = umg_data[spec][0].solution
        ds_sol_err[lab] = umg_data[spec][1]

    # make style iterables
    colors = ['green', 'red', 'blue', 'orange', 'black']
    linestyles = ['-', '-.', ':', '--', '-']
    markers = ['o', 's', '^', '*']

    # FOR DEFAULT SPECTRA
    # setup plotting environment
    fig = plt.figure(1234)
    ax = fig.add_subplot(111)
    ax.set_xlabel('E $MeV$')
    ax.set_ylabel('$\Phi$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plot default spectra
    ax.plot(*ts.step, label='True', linestyle='-', linewidth=1.0, color='k')
    for i, val in enumerate(ds.items()):
        lab, spec = val
        ax.plot(*spec.step, label=lab, linestyle=linestyles[i], linewidth=1.0, color=colors[i])

    ax.legend(frameon=False)
    fig.savefig(plot_path + 'umg_ds_{}.pdf'.format(pgm))
    fig.clf()

    # FOR DEFAULT SPECTRA SOLUTIONS
    # setup plotting environment
    fig = plt.figure(1235)
    ax = fig.add_subplot(111)
    ax.set_xlabel('E $MeV$')
    ax.set_ylabel('$\Phi$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plot spectra
    ax.plot(*ts.step, label='True', linestyle='-', linewidth=1.0, color='k')
    for i, val in enumerate(ds_sol.items()):
        lab, spec = val
        ax.plot(*spec.step, label=lab, linestyle=linestyles[i], linewidth=1.0, color=colors[i])

    ax.legend(frameon=False)
    fig.savefig(plot_path + 'umg_ds_sol_{}.pdf'.format(pgm))
    fig.clf()

    # FOR DEFAULT SPECTRA SOLUTION ERRORS
    # setup plotting environment
    fig = plt.figure(1236)
    ax = fig.add_subplot(111)
    ax.set_xlabel('g')
    ax.set_ylabel('%')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, val in enumerate(ds_sol_err.items()):
        lab, err = val
        ax.plot(range(len(err)), err, label=lab + ' ({:4.1f})'.format(sp.mean(err)),
                linestyle=linestyles[i], linewidth=1.0, color=colors[i], marker=markers[i])

    ax.legend(frameon=False)
    fig.savefig(plot_path + 'umg_ds_sol_err_{}.pdf'.format(pgm))
    fig.clf()


make_ds_plot(umg_data, 'mx')
make_ds_plot(umg_data, 'gr')
