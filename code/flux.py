import numpy as np
from scipy.interpolate import interp1d


def select_flux_spectrum(flux, power):
    message = 'Not a valid flux spectrum.'
    assert flux in ['trigaA', 'trigaB', 'trigaC', 'trigaD', 'trigaE', 'trigaF', 'triga_nebp'], message

    if 'triga' in flux and len(flux) == 6:
        port = flux[-1]

        data = np.load('mpfd_nodal_data.npy')[()]

        nodal_spectra = [-1]
        for i in range(1, 5):
            node_data = data[port + str(i)]
            erg, values, err = node_data.T
            midpoints = (erg[1:] + erg[:-1]) / 2
            widths = erg[1:] - erg[:-1]
            values = (values[1:] / widths) * power
            nodal_spectra.append(interp1d(midpoints, values, bounds_error=False, fill_value=0))
        return nodal_spectra

    elif flux == 'triga_nebp':
        data = np.load('nebp_data.npy')
        nodal_spectra = [-1]
        erg, values, err = data
        midpoints = (erg[1:] + erg[:-1]) / 2
        widths = erg[1:] - erg[:-1]
        values = (values[1:] / widths) * power
        for i in range(1, 5):
            nodal_spectra.append(interp1d(midpoints, values, bounds_error=False, fill_value=0))
        return nodal_spectra
