import numpy as np
from scipy.interpolate import interp1d as interp

from master_data import directory, img_directory

def load_cross_sections(isos) :
    """Read ascii cross sections and return as a dictionary."""
    data = {}
    for iso in isos :
        E, sig = np.loadtxt(directory+'/data/'+iso+'.txt', skiprows=1, 
                            delimiter=',', unpack=True) 
        data[iso] = {}
        data[iso]['value'] = sig
        data[iso]['E'] = E
    return data
    
def get_cross_section_interps(isos) :
    data = load_cross_sections(isos)
    interps = {}
    for iso in isos :
        E, val = data[iso]['E'], data[iso]['value']
        left, right = data[iso]['value'][0], data[iso]['value'][-1]
        interps[iso] = interp(E, val, bounds_error=False, fill_value=(left, right))
    return interps
    
if __name__ == "__main__" :
    from nice_plots import init_nice_plots
    init_nice_plots()
    import matplotlib.pyplot as plt
    from master_data import isos, isos_str, isos_colors
    data = load_cross_sections(isos)
    interps = get_cross_section_interps(isos)
    E = np.logspace(-5, np.log10(2e7), int(1e5))
    for iso in  isos :
        plt.loglog(E, interps[iso](E), label=isos_str[iso], color=isos_colors[iso])
    plt.legend(loc=0, ncol=4)
    plt.axis([1e-5, 1e7, 1e-11, 1e5])
    plt.xlabel('E (eV)')
    plt.ylabel('$\sigma_f$ (b)')
    plt.savefig(img_directory+'fission_cross_sections.pdf')  
    #plt.show()
    
    
   