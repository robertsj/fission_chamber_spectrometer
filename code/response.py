from multigroup_utilities import energy_groups
from scipy.integrate import quad, trapz
from process_cross_sections import get_cross_section_interps
import scipy as sp
import os
import pickle

def generate_responses(isos, flux, struct='wims69', lower=1e-5, upper=2e7, 
                       name='response.p', overwrite=False) :
    """This generates response functions for given isotopes,
     a given spectrum, and a desired group structure."""
    if os.path.isfile(name) and not overwrite :
        return pickle.load(open(name, 'rb')) 
    responses = {}
    eb = energy_groups(struct, lower, upper)
    responses['eb'] = eb
    responses['phi'] = sp.zeros(len(eb)-1)
    for i in range(len(eb)-1) :
        E = sp.logspace(sp.log10(eb[i+1]), sp.log10(eb[i]), 1e4)
        x = trapz(flux(E), E)
        print("->",i,eb[i+1], eb[i], x)
        responses['phi'][i] = trapz(flux(E), E)
        
    interps = get_cross_section_interps(isos)
    responses['response'] = {}
    for iso in isos :
        print("...response for iso ", iso)
        responses['response'][iso] = sp.zeros(len(eb)-1)
        fun = lambda x: flux(x)*interps[iso](x)
        for i in range(len(eb)-1) :
            E = sp.logspace(sp.log10(eb[i+1]), sp.log10(eb[i]), 1e4)
            top = trapz(fun(E), E)
            responses['response'][iso][i] = top/responses['phi'][i]
    pickle.dump(responses, open(name, 'wb'))
    return responses
    
if __name__ == "__main__" :
    from master_data import isos
    from flux_spectrum import Flux
    from multigroup_utilities import plot_multigroup_data
    struct = 'wims69'
    phi = Flux(7.0, 600.0)
    resp = generate_responses(isos, phi.evaluate, name='test_resp.p', overwrite=True)
    for iso in isos:
        x, y = plot_multigroup_data(resp['eb'], resp['response'][iso])
        plt.loglog(x, y, ls='-')