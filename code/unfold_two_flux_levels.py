import pickle
import scipy as sp
from scipy.optimize import fmin_cobyla, minimize

def integral_response(name, v = 1):
    response = pickle.load(open(name, 'rb'))
    phi = response['phi']
    phi = phi / sum(phi) * v
    R = {}
    for iso in response['response'].keys() :
        R[iso] = phi.dot(response['response'][iso])
    return R

def entropy(v, grad = None) :
    return -sum(v*sp.log(v)) # - entropy to 

def equal_1(x, R, RF, isos, grad=None) :
        delta = np.zeros(len(isos))
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
         
def unfold(R1, R2, RF, isos) :
    """ Unfold the spectrum given the integral responses, response functions,
        and isotopes to use. """
    objective = lambda x: -entropy(x, None)
    eq1 = {'type': 'eq', 'fun': lambda x: equal_1(x, R1, RF, isos)}
    eq2 = {'type': 'eq', 'fun': lambda x: sum(x)-1}
    phi0 = np.ones(len(RF['u235']))
    phi0 = phi0 / sum(phi0)    
    bounds = [(1e-13, 1.0)]*len(phi0)
    y = minimize(objective, phi0, constraints=[eq1, eq2], method='SLSQP', bounds=bounds) 
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

    R1 = integral_response(name, 1)
    R2 = integral_response(name, 2)
    RF = resp['response']

    phi_ref = resp['phi']
    phi_ref = phi_ref / sum(phi_ref)
    eb = resp['eb']
    
    #y3 = unfold(R, RF, isos=['u235','u238','th232'])
    phi_3 = unfold(R1, R2, RF, isos=['u235', 'u238', 'th232'])
    phi_5 = unfold(R1, R2, RF, isos=['u235', 'u238', 'th232', 'np237', 'pu238'])
    phi_11 = unfold(R1, R2, RF, isos=['th232','u233','u234','u235','u238','np237','pu238','pu239','pu240','pu241','pu242'])
 
    
    x,yr = plot_multigroup_data(eb, phi_ref, 'group-to-e')
    x,ur = plot_multigroup_data(eb, phi_ref, 'group-to-u')
    x,y3 = plot_multigroup_data(eb, phi_3, 'group-to-e')
    x,y5 = plot_multigroup_data(eb, phi_5, 'group-to-e')
    x,y11 = plot_multigroup_data(eb, phi_11, 'group-to-e')
   # x,y15 = plot_multigroup_data(eb, phi_15, 'group-to-e')
   
    plt.figure(1)
    plt.loglog(x, yr, 'k', x, y3, 'b--', x, y5, 'g-.', x, y11, 'r:')
    
        
    plt.figure(2)
    plt.plot(range(1,1+len(phi_3)), phi_3/phi_ref, 'b--s',
             range(1,1+len(phi_5)), phi_5/phi_ref, 'g-.o', 
             range(1, 1+len(phi_5)), phi_11/phi_ref, 'r:*')