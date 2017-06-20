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
        total flux, and isotopes to use. """
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

def unfold_no_norm(R, RF, isos) :
    """ Unfold the spectrum given the integral responses, response functions,
        and isotopes to use. """
    objective = lambda x: -entropy(x, None)
    eq1 = {'type': 'eq', 'fun': lambda x: equal_1(x, R, RF, isos)}
    phi0 = sp.ones(len(RF['u235']))
    phi0 = phi0 / sum(phi0) 
    bounds = [(1e-13, 1.0)]*len(phi0)
    y = minimize(objective, phi0, constraints=[eq1], method='SLSQP', bounds=bounds) 
    print("objective = %12.5e" % y.fun)
    #print(y)
    return y.x

def unfold_min_norm(R, RF, isos) :
    """ Unfold using pseudo-inverse approach."""
    A = sp.zeros((len(isos), len(RF['u235'])))
    b = sp.zeros(len(isos))
    k = 0
    for iso in isos :
        A[k, :] = RF[iso]
        b[k] = R[iso]
        k += 1
    M = A.dot(A.T) 
    print(M.shape)
    #A = RF.dot(RF.transpose())
    #w = sp.linalg.solve(A, R)
   # Phi_g = RF.transpose().dot(w) 
    y = sp.linalg.solve(A.dot(A.T), b)
    y = A.T.dot(y)
    return y

if __name__ == "__main__" :
    from flux_spectrum import Flux
    from master_data import isos
    from response import generate_responses
    from multigroup_utilities import energy_groups, plot_multigroup_data
    import matplotlib.pyplot as plt
    from nice_plots import init_nice_plots
    init_nice_plots()
    
    struct = 'wims69'
    pwr = Flux(7.0, 600.0)
    name = 'test_wims69_resp.p'
    resp = generate_responses(isos, pwr.evaluate, struct=struct, name=name, overwrite=False)
    R = integral_response(name)
    RF = resp['response']

    phi_ref = resp['phi']
    phi_ref = phi_ref / sum(phi_ref)
    eb = resp['eb']
    
    isos_1 = ['u235', 'u238', 'th232']
    isos_2 = ['u235', 'u238', 'th232', 'np237', 'pu238']
    isos_3 = ['th232','u233','u234','u235','u238','np237',
              'pu238','pu239','pu240','pu241','pu242']
    isos_4 = isos
    
    #y3 = unfold(R, RF, isos=['u235','u238','th232'])
    phi_3 = unfold(R, RF, isos=isos_1)
    phi_5 = unfold(R, RF, isos=isos_2)
    phi_11 = unfold(R, RF, isos=isos_3)
    phi_15 = unfold(R, RF, isos=isos_4)
    phi_15_phi0_5 = unfold(R, RF, isos=isos_4, tot=0.5)
    phi_15_phi2_0 = unfold(R, RF, isos=isos_4, tot=2.0)
    phi_3_min_norm = unfold_min_norm(R, RF, isos=isos_1)
    phi_15_min_norm = unfold_min_norm(R, RF, isos=isos_4)
    
    x,yr = plot_multigroup_data(eb, phi_ref, 'group-to-e')
    x,ur = plot_multigroup_data(eb, phi_ref, 'group-to-u')
    x,y3 = plot_multigroup_data(eb, phi_3, 'group-to-e')
    x,u3 = plot_multigroup_data(eb, phi_3, 'group-to-u')
    x,y5 = plot_multigroup_data(eb, phi_5, 'group-to-e')
    x,u5 = plot_multigroup_data(eb, phi_5, 'group-to-u')
    x,y11 = plot_multigroup_data(eb, phi_11, 'group-to-e')
    x,u11 = plot_multigroup_data(eb, phi_11, 'group-to-u')
    x,y15 = plot_multigroup_data(eb, phi_15, 'group-to-e')
    x,u15 = plot_multigroup_data(eb, phi_15, 'group-to-u')

    x,y3_mn = plot_multigroup_data(eb, phi_3_min_norm, 'group-to-e')
    x,u3_mn = plot_multigroup_data(eb, phi_3_min_norm, 'group-to-u')    
    x,y15_mn = plot_multigroup_data(eb, phi_15_min_norm, 'group-to-e')
    x,u15_mn = plot_multigroup_data(eb, phi_15_min_norm, 'group-to-u')

    """ Reconstructed flux spectra for sum(phi) = 1 and several sets of nuclides """
    plt.figure(1)
    plt.loglog(x, yr, 'k', x, y3, 'b--', x, y5, 'g-.', x, y11, 'r:', x, y15, 'c--')
    plt.xlabel('$E$ (eV)')
    plt.ylabel('$\phi(E)$')
    plt.legend(['reference', 'case 1', 'case 2', 'case 3', 'case 4'], loc=0)
    plt.savefig('reconstructed_flux.pdf')
    
    """ Reconstructed flux spectra per unit lethargy """
    plt.figure(2)
    plt.loglog(x, ur, 'k', x, u3, 'b--', x, u5, 'g-.', x, u11, 'r:', x, u15, 'c--')
    plt.legend('$E$ (eV)')
    plt.ylabel('$E \phi(E)$')
    plt.legend(['reference', 'case 1', 'case 2', 'case 3', 'case 4'], loc=0)
    plt.savefig('reconstructed_flux_lethargy.pdf')
    

    """ Group-wise, relative error """   
    err3, err5, err11, err15 =sp.mean(100*abs(phi_3-phi_ref)/phi_ref),\
        sp.mean(100*abs(phi_5-phi_ref)/phi_ref),\
        sp.mean(100*abs(phi_11-phi_ref)/phi_ref),\
        sp.mean(100*abs(phi_15-phi_ref)/phi_ref)
        
    print("Average group-wise errors")
    print(sp.mean(100*abs(phi_3-phi_ref)/phi_ref))
    print(sp.mean(100*abs(phi_5-phi_ref)/phi_ref))
    print(sp.mean(100*abs(phi_11-phi_ref)/phi_ref))
    print(sp.mean(100*abs(phi_15-phi_ref)/phi_ref))
    
    plt.figure(3)
    plt.plot(range(1,1+len(phi_3)), 100*abs(phi_3-phi_ref)/phi_ref, 'b--s',
             range(1,1+len(phi_5)), 100*abs(phi_5-phi_ref)/phi_ref, 'g-.o', 
             range(1, 1+len(phi_5)), 100*abs(phi_11-phi_ref)/phi_ref, 'r:*',
             range(1, 1+len(phi_5)), 100*abs(phi_15-phi_ref)/phi_ref, 'c--h')
    plt.legend(['case 1 (%.1f)' % err3, 
                'case 2 (%.1f)' % err5, 
                'case 3 (%.1f)' % err11,  
                'case 4 (%.1f)' % err15], loc=0)
    plt.savefig('groupwise_error.pdf')

    """ Group-wise, relative error zoomed"""    
    plt.figure(4)
    plt.plot(range(1,1+len(phi_3)), 100*abs(phi_3-phi_ref)/phi_ref, 'b--s',
             range(1,1+len(phi_5)), 100*abs(phi_5-phi_ref)/phi_ref, 'g-.o', 
             range(1, 1+len(phi_5)), 100*abs(phi_11-phi_ref)/phi_ref, 'r:*',
             range(1, 1+len(phi_5)), 100*abs(phi_15-phi_ref)/phi_ref, 'c--h')
    plt.legend(['case 1 (%.1f)' % err3, 
                'case 2 (%.1f)' % err5, 
                'case 3 (%.1f)' % err11,  
                'case 4 (%.1f)' % err15], loc=0)
    plt.axis([0, 70, 0, 100])
    plt.savefig('groupwise_error_zoomed.pdf')
    

    
    plt.figure(5)
    err1_0, err0_5, err2_0 = \
        sp.mean(100*abs(phi_15-phi_ref)/phi_ref), \
        sp.mean(100*abs(phi_15_phi0_5-phi_ref)/phi_ref), \
        sp.mean(100*abs(phi_15_phi2_0-phi_ref)/phi_ref)
    plt.plot(range(1, 1+len(phi_5)), 100*abs(phi_15-phi_ref)/phi_ref, 'b--s',
             range(1, 1+len(phi_5)), 100*abs(phi_15_phi0_5-phi_ref)/phi_ref, 'g-.o',
             range(1, 1+len(phi_5)), 100*abs(phi_15_phi2_0-phi_ref)/phi_ref, 'r:*')
    plt.legend(['$\sum \phi = 1.0$ (%.1f)' % err1_0, 
                '$\sum \phi = 0.5$ (%.1f)' % err0_5, 
                '$\sum \phi = 2.0$ (%.1f)' % err2_0], loc=0)
    plt.savefig('different_total_fluxes.pdf')
    #wplt.axis([0, 70, 0, 200])
    
   
    plt.figure(6)
    plt.title("Fluxes")
    plt.loglog(x, yr, 'k', x, y15, 'r:', x, y15_mn, 'c--')# x, y_11_phi1_5, 'g-.', x, y_11_phi2_0, 'r:')
   # plt.legend(['ref', '1.0', '1.5', '2.0'])
    
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