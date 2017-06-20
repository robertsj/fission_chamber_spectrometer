import pickle
import scipy as sp
from scipy.optimize import fmin_cobyla, minimize

def integral_response(name):
    response = pickle.load(open(name, 'rb'))
    phi = response['phi']
    phi = phi / sum(phi) 
    R = {}
    for iso in response['response'].keys() :
        R[iso] = phi.dot(response['response'][iso])
    return R

def entropy(v, grad = None) :
    return -sum(v*sp.log(v)) # - entropy to 

def equal_1(x, R, RF, isos, grad=None) :
        delta = sp.zeros(len(isos))
        RR = []
        for i in range(len(isos)):
            iso = isos[i]
            delta[i] = (RF[iso].dot(x) - R[iso])/R[iso]
            RR.append(RF[iso].dot(x))
        #print delta
        #for i in range(len(RR)) :
        #    print '  ....%4i %12.8f, %12.8f, %12.3e, %12.3e    ' \
        #      % ( i, RR[i], R[data['isos'][i]], RR[i]- R[data['isos'][i]], delta[i])
        return delta
         
def unfold(R, RF, isos, tot = 1) :
    """ Unfold the spectrum given the integral responses, response functions,
        and isotopes to use. """
    objective = lambda x: -entropy(x, None)
    eq1 = {'type': 'eq', 'fun': lambda x: equal_1(x, R, RF, isos)}
    eq2 = {'type': 'eq', 'fun': lambda x: tot-sum(x)}
    phi0 = sp.ones(len(RF['u235']))
    phi0 = phi0 / sum(phi0) 
    bounds = [(1e-13, 1.0)]*len(phi0)
    y = minimize(objective, phi0, constraints=[eq1, eq2], method='SLSQP', bounds=bounds) 
    print("objective = %12.5e" % y.fun)
    #print(y)
    return y.x


if __name__ == "__main__" :
    from flux_spectrum import Flux
    from master_data import isos
    from response import generate_responses
    from multigroup_utilities import energy_groups, plot_multigroup_data
    import matplotlib.pyplot as plt
    
    struct = 'wims69'
    pwr = Flux(7.0, 600.0)
    name = 'test_wims69_resp.p'
    resp = generate_responses(isos, pwr.evaluate, struct=struct, name=name, overwrite=False)
    R = integral_response(name)
    RF = resp['response']

    phi_ref = resp['phi']
    phi_ref = phi_ref / sum(phi_ref)
    eb = resp['eb']
    
    #y3 = unfold(R, RF, isos=['u235','u238','th232'])
    phi_3 = unfold(R, RF, isos=['u235', 'u238', 'th232'])
    phi_5 = unfold(R, RF, isos=['u235', 'u238', 'th232', 'np237', 'pu238'])
    phi_11 = unfold(R, RF, isos=['th232','u233','u234','u235','u238','np237',
                    'pu238','pu239','pu240','pu241','pu242'], tot=1.0)
    phi_11_phi0_5 = unfold(R, RF, isos=['th232','u233','u234','u235','u238','np237',
                           'pu238','pu239','pu240','pu241','pu242'], tot=0.5)
    phi_11_phi2_0 = unfold(R, RF, isos=['th232','u233','u234','u235','u238','np237',
                           'pu238','pu239','pu240','pu241','pu242'], tot=2.0)
    phi_15 = unfold(R, RF, isos=['u235', 'u238', 'th232', 'np237', 'pu238', 'b10', 'li6', 'ni59'])
 
    
    x,yr = plot_multigroup_data(eb, phi_ref, 'group-to-e')
    x,ur = plot_multigroup_data(eb, phi_ref, 'group-to-u')
    x,y3 = plot_multigroup_data(eb, phi_3, 'group-to-e')
    x,u3 = plot_multigroup_data(eb, phi_3, 'group-to-u')
    x,y5 = plot_multigroup_data(eb, phi_5, 'group-to-e')
    x,u5 = plot_multigroup_data(eb, phi_5, 'group-to-u')
    print('------>', phi_11[0:3])  
    x,y11 = plot_multigroup_data(eb, phi_11, 'group-to-e')
    print('------>', phi_11[0:3])  
    x,u11 = plot_multigroup_data(eb, phi_11, 'group-to-u')
    print('------>', phi_11[0:3])  
    x,y15 = plot_multigroup_data(eb, phi_15, 'group-to-e')
    x,u15 = plot_multigroup_data(eb, phi_15, 'group-to-u')

    """ Reconstructed flux spectra for sum(phi) = 1 and several sets of nuclides """
    plt.figure(1)
    plt.title(1)
    plt.loglog(x, yr, 'k', x, y3, 'b--', x, y5, 'g-.', x, y11, 'r:', x, y15, 'c--')
    plt.xlabel('$E$ (eV)')
    plt.ylabel('$\phi(E)$')
    plt.legend(['ref', '3', '5', '11', '15'])
    
    """ Reconstructed flux spectra per unit lethargy """
    plt.figure(2)
    plt.title(2)
    plt.loglog(x, ur, 'k', x, u3, 'b--', x, u5, 'g-.', x, u11, 'r:', x, u15, 'c--')
    plt.legend('$E$ (eV)')
    plt.ylabel('$E \phi(E)$')
    plt.legend(['ref', '3', '5', '11', '15'])

    """ Group-wise, relative error """    
    plt.figure(3)
    plt.title("Relative Error")
    plt.plot(range(1,1+len(phi_3)), 100*abs(phi_3-phi_ref)/phi_ref, 'b--s',
             range(1,1+len(phi_5)), 100*abs(phi_5-phi_ref)/phi_ref, 'g-.o', 
             range(1, 1+len(phi_5)), 100*abs(phi_11-phi_ref)/phi_ref, 'r:*',
             range(1, 1+len(phi_5)), 100*abs(phi_15-phi_ref)/phi_ref, 'c--h')
    plt.legend(['3', '5', '11', '15'])
    
    """ Group-wise, relative error zoomed"""    
    plt.figure(4)
    plt.title("Relative Error")
    plt.plot(range(1,1+len(phi_3)), 100*abs(phi_3-phi_ref)/phi_ref, 'b--s',
             range(1,1+len(phi_5)), 100*abs(phi_5-phi_ref)/phi_ref, 'g-.o', 
             range(1, 1+len(phi_5)), 100*abs(phi_11-phi_ref)/phi_ref, 'r:*',
             range(1, 1+len(phi_5)), 100*abs(phi_15-phi_ref)/phi_ref, 'c--h')
    plt.legend(['3', '5', '11', '15'])
    plt.axis([0, 70, 0, 100])
    
    
    plt.figure(5)
    plt.title("Relative Error")
    plt.plot(range(1, 1+len(phi_5)), 100*abs(phi_11-phi_ref)/phi_ref, 'b--s',
             range(1, 1+len(phi_5)), 100*abs(phi_11_phi0_5-phi_ref)/phi_ref, 'g-.o',
             range(1, 1+len(phi_5)), 100*abs(phi_11_phi2_0-phi_ref)/phi_ref, 'r:*')
    plt.legend(['1.0', '0.5', '2.0'])
    #wplt.axis([0, 70, 0, 200])
    
    #_,y_11 = plot_multigroup_data(eb, phi_11, 'group-to-e')
    #_,y_11_phi1_5 = plot_multigroup_data(eb, phi_11_phi1_5, 'group-to-e')
    #_,y_11_phi2_0 = plot_multigroup_data(eb, phi_11_phi2_0, 'group-to-e')

    plt.figure(6)
    plt.title("Fluxes")
    plt.loglog(x, yr, 'k', x, y11, 'r:')# x, y_11_phi1_5, 'g-.', x, y_11_phi2_0, 'r:')
    plt.legend(['ref', '1.0', '1.5', '2.0'])
    
    """ Conclusion: we don't need to have exact total flux, but we do need to
        normalize it to something reasonable as part of the constraints.
        
        For talk:
            - show results with 0.5, 1.0, 2.0 flux and the result without
              normalization
            - show table of sensitivities for the given flux spectrum
              (maybe grouped?)
            -
    #wplt.axis([0, 70, 0, 200])
    """