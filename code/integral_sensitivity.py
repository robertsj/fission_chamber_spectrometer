from process_cross_sections import get_cross_section_interps
from scipy.integrate import quad, trapz
def compute_integral_sensitivities(isos, phi, low=1e-5, high=2e7) :
    """Compute the integral sensitivity using an assumed spectrum."""
    interps = get_cross_section_interps(isos)
    E = np.logspace(np.log10(low), np.log10(high), 4e5)
    Eth = np.logspace(np.log10(low), np.log10(0.625), 4e5)
    Ef = np.logspace(np.log10(0.625), np.log10(high), 4e5) 
    bot = trapz(phi.evaluate(E), E)
    bot_th = trapz(phi.evaluate(Eth), Eth)
    bot_f = trapz(phi.evaluate(Ef), Ef)
    print(bot, bot_th, bot_f)
    sens = []
    for iso in isos :
        top = trapz(interps[iso](E)*phi.evaluate(E), E)
        top_th = trapz(interps[iso](Eth)*phi.evaluate(Eth), Eth)
        top_f = trapz(interps[iso](Ef)*phi.evaluate(Ef), Ef)
        sens.append((iso, top/bot, top_th/bot_th, top_f/bot_f))
    return sorted(sens, key=lambda x: -x[1])
                
if __name__ == "__main__" :
    import numpy as np

    from master_data import isos, isos_str
    from flux_spectrum import Flux
    pwr = Flux(7.0, 600.0)
    sens = compute_integral_sensitivities(isos, pwr)
    for s in sens :
        S = list(s)
        S[0] = isos_str[S[0]]
        S=tuple(S)
        print(r" %s & %.4f &  %.4f &  %.4f\\" % S)

