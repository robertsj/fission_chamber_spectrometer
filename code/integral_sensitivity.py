from process_cross_sections import get_cross_section_interps
from scipy.integrate import quad, trapz
def compute_integral_sensitivities(isos, phi, low=1e-5, high=2e7) :
    """Compute the integral sensitivity using an assumed spectrum."""
    interps = get_cross_section_interps(isos)
    E = np.logspace(np.log10(low), np.log10(high), 4e5)
    bot = trapz(phi.evaluate(E), E)
    sens = []
    for iso in isos :
        top = trapz(interps[iso](E)*phi.evaluate(E), E)
        sens.append((iso, top/bot))
    return sorted(sens, key=lambda x: -x[1])
                
if __name__ == "__main__" :
    import numpy as np

    from master_data import isos, isos_str
    from flux_spectrum import Flux
    pwr = Flux(7.0, 600.0)
    sens = compute_integral_sensitivities(isos, pwr)
    for s in sens :
        print(s[0])#, "%12.4f" % s[1]) 

