import matplotlib.pyplot as plt
from master_data import isos, isos_str, isos_colors
from flux_spectrum import Flux
from multigroup_utilities import plot_multigroup_data
from response import generate_responses
import numpy as np
def plot_cross_sections() :
    data = load_cross_sections(isos)
    interps = get_cross_section_interps(isos)
    E = np.logspace(-5, np.log10(2e7), 1e5)
    for iso in  isos :
        plt.loglog(E, interps[iso](E), label=isos_str[iso], color=isos_colors[iso])
    plt.legend()
    plt.show()

def generate_all_responses() :
    
    # 2 group
    struct = 'tg0_625'
    fun = lambda x: 0*x+1
    resp = generate_responses(isos, fun, struct=struct, name='tg0_625_flat_resp.p', overwrite=True)
    pwr = Flux(7.0, 600.0)
    resp = generate_responses(isos, pwr.evaluate, struct=struct, name='tg0_625_pwr_resp.p', overwrite=True)
    triga = Flux(2.0, 600.0) 
    resp = generate_responses(isos, triga.evaluate, struct=struct, name='tg0_625_triga_resp.p', overwrite=True)
  
    
    # 32 group
    struct = 'phoenix25'
    fun = lambda x: 0*x+1
    resp = generate_responses(isos, fun, struct=struct, name='phoenix25_flat_resp.p', overwrite=True)
    pwr = Flux(7.0, 600.0)
    resp = generate_responses(isos, pwr.evaluate, struct=struct, name='phoenix25_pwr_resp.p', overwrite=True)
    triga = Flux(2.0, 600.0) 
    resp = generate_responses(isos, triga.evaluate, struct=struct, name='phoenix25_triga_resp.p', overwrite=True)
  
    
    
    
    # 69 group
    struct = 'wims69'
    fun = lambda x: 0*x+1
    resp = generate_responses(isos, fun, struct=struct, name='wims69_flat_resp.p', overwrite=True)
    pwr = Flux(7.0, 600.0)
    resp = generate_responses(isos, pwr.evaluate, struct=struct, name='wims69_pwr_resp.p', overwrite=True)
    triga = Flux(2.0, 600.0) 
    resp = generate_responses(isos, triga.evaluate, struct=struct, name='wims69_triga_resp.p', overwrite=True)
  
    
    # 238 group
    struct = 'scale238'
    fun = lambda x: 0*x+1
    resp = generate_responses(isos, fun, struct=struct, name='wims238_flat_resp.p', overwrite=True)
    pwr = Flux(7.0, 600.0)
    resp = generate_responses(isos, pwr.evaluate, struct=struct, name='wims238_pwr_resp.p', overwrite=True)
    triga = Flux(2.0, 600.0) 
    resp = generate_responses(isos, triga.evaluate, struct=struct, name='wims238_triga_resp.p', overwrite=True)
  
    
    
if __name__ == "__main__" :
    
    generate_all_responses()
    
    resp1 = generate_responses(isos, None, name='tg0_625_flat_resp.p', overwrite=False)
    resp2 = generate_responses(isos, None, name='tg0_625_pwr_resp.p', overwrite=False)

    for iso in isos:
        x, y = plot_multigroup_data(resp1['eb'],  resp2[iso])
        plt.loglog(x, y, ls='-',color=isos_colors[iso])
        #plt.legend(isos)