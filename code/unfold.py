import pickle
import scipy as sp
from scipy.optimize import fmin_cobyla, minimize
from master_data import directory

def integral_response(name):
    response = pickle.load(open(directory+'/code/'+name, 'rb'))
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

def unfold_tikhonov(R, RF, isos, alpha) :
    """ Unfold using tikhonov regularization."""
    A = sp.zeros((len(isos), len(RF['u235'])))
    b = sp.zeros(len(isos))
    k = 0
    for iso in isos :
        A[k, :] = RF[iso]
        b[k] = R[iso]
        k += 1
    M = A.T.dot(A) + alpha**2*sp.eye(len(A.T))
    print(M.shape)
    #A = RF.dot(RF.transpose())
    #w = sp.linalg.solve(A, R)
   # Phi_g = RF.transpose().dot(w) 
    y = sp.linalg.solve(M, A.T.dot(b))
    #y = A.T.dot(y)
    return y

if __name__ == "__main__" :
    from flux_spectrum import Flux
    from master_data import isos, img_directory
    from response import generate_responses
    from multigroup_utilities import energy_groups, plot_multigroup_data
    import matplotlib.pyplot as plt
    from nice_plots import init_nice_plots
    init_nice_plots()
    
    
    isos_cd = ['u233cd113', 'u235cd113', 'pu238cd113', 'pu239cd113', 'pu241cd113']
    isos_gd = ['u233gd155', 'u235gd155', 'pu238gd155', 'pu239gd155', 'pu241gd155']
  
    
    struct = 'wims69'
    pwr = Flux(7.0, 600.0)
    name = 'test_wims69_resp.p'
    resp = generate_responses(isos+isos_cd+isos_gd, pwr.evaluate, 
                              struct=struct, name=name, overwrite=True)
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
    

    isos_5 = isos_3 + isos_cd
    isos_6 = isos_3 + isos_gd
    isos_7 = isos_3 + isos_cd + isos_gd
    
    #y3 = unfold(R, RF, isos=['u235','u238','th232'])
    phi_3 = unfold(R, RF, isos=isos_1)
    phi_5 = unfold(R, RF, isos=isos_2)
    phi_11 = unfold(R, RF, isos=isos_3)
    phi_15 = unfold(R, RF, isos=isos_4)
    phi_15_phi0_5 = unfold(R, RF, isos=isos_4, tot=0.5)
    phi_15_phi2_0 = unfold(R, RF, isos=isos_4, tot=2.0)
    phi_3_min_norm = unfold_min_norm(R, RF, isos=isos_1)
    phi_15_min_norm = unfold_min_norm(R, RF, isos=isos_4)
    phi_15_tik_0_1 = unfold_tikhonov(R, RF, isos=isos_4, alpha=0.1)
    phi_15_tik_1_0 = unfold_tikhonov(R, RF, isos=isos_4, alpha=0.01)
    
    phi_cd =  unfold(R, RF, isos=isos_5, tot=1.0)
    phi_gd =  unfold(R, RF, isos=isos_6, tot=1.0)
    phi_cdgd =  unfold(R, RF, isos=isos_7, tot=1.0)
    
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
    x,y15_ti_0_1 = plot_multigroup_data(eb, phi_15_tik_0_1, 'group-to-e')
    x,u15_ti_0_1 = plot_multigroup_data(eb, phi_15_tik_0_1, 'group-to-u')
    x,y15_ti_1_0 = plot_multigroup_data(eb, phi_15_tik_1_0, 'group-to-e')
    x,u15_ti_1_0 = plot_multigroup_data(eb, phi_15_tik_1_0, 'group-to-u')
    
    x,ycd = plot_multigroup_data(eb, phi_cd, 'group-to-e')
    x,ucd = plot_multigroup_data(eb, phi_cd, 'group-to-u') 
    x,ygd = plot_multigroup_data(eb, phi_gd, 'group-to-e')
    x,ugd = plot_multigroup_data(eb, phi_gd, 'group-to-u') 
    x,ycdgd = plot_multigroup_data(eb, phi_cdgd, 'group-to-e')
    x,ucdgd = plot_multigroup_data(eb, phi_cdgd, 'group-to-u') 
    
    """ Reconstructed flux spectra for sum(phi) = 1 and several sets of nuclides """
    plt.figure(1)#, figsize=(12, 7))
    plt.loglog(x, yr, 'k', x, y3, 'b--', x, y5, 'g-.', x, y11, 'r:', x, y15, 'c--')
    plt.xlabel('$E$ (eV)')
    plt.ylabel('$\phi(E)$')
    plt.legend(['reference', 'case 1', 'case 2', 'case 3', 'case 4'], loc=0)
    #plt.grid(True, alpha=0.5)
    plt.savefig(img_directory+'reconstructed_flux.pdf')
    
    """ Reconstructed flux spectra per unit lethargy """
    plt.figure(2)
    plt.loglog(x, ur, 'k', x, u3, 'b--', x, u5, 'g-.', x, u11, 'r:', x, u15, 'c--')
    plt.xlabel('$E$ (eV)')
    plt.ylabel('$E \phi(E)$')
    plt.legend(['reference', 'case 1', 'case 2', 'case 3', 'case 4'], loc=0)
    plt.savefig(img_directory+'reconstructed_flux_lethargy.pdf')
    

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
    plt.xlabel('g')
    plt.savefig(img_directory+'groupwise_error.pdf')

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
    plt.xlabel('g')
    plt.savefig(img_directory+'groupwise_error_zoomed.pdf')
    

    
    plt.figure(5)
    err1_0, err0_5, err2_0 = \
        sp.mean(100*abs(phi_15-phi_ref)/phi_ref), \
        sp.mean(100*abs(phi_15_phi0_5-phi_ref)/phi_ref), \
        sp.mean(100*abs(phi_15_phi2_0-phi_ref)/phi_ref)
    plt.plot(range(1, 1+len(phi_5)), 100*abs(phi_15-phi_ref)/phi_ref, 'b--s',
             range(1, 1+len(phi_5)), 100*abs(phi_15_phi0_5-phi_ref)/phi_ref, 'g-.o',
             range(1, 1+len(phi_5)), 100*abs(phi_15_phi2_0-phi_ref)/phi_ref, 'r:*')
    plt.legend(['$\phi_{tot} = 1.0$ (%.1f)' % err1_0, 
                '$\phi_{tot} = 0.5$ (%.1f)' % err0_5, 
                '$\phi_{tot} = 2.0$ (%.1f)' % err2_0], loc=0)
    plt.savefig(img_directory+'different_total_fluxes.pdf')
    #wplt.axis([0, 70, 0, 200])
    
   
    plt.figure(6)
    err_me = sp.mean(100*abs(phi_15-phi_ref)/phi_ref)
    err_mr = sp.mean(100*abs(phi_15_min_norm-phi_ref)/phi_ref)
    err_ti_0_1 = sp.mean(100*abs(phi_15_tik_0_1-phi_ref)/phi_ref)
    err_ti_1_0 = sp.mean(100*abs(phi_15_tik_1_0-phi_ref)/phi_ref)
    plt.loglog(x, yr, 'k', x, y15, 'r:', x, y15_mn, 'c--',x, y15_ti_0_1, 'b-.',x, y15_ti_1_0, 'g--')
    plt.legend(['reference', 
                'max. entropy (%.1f)' % err_me, 
                'min. norm (%.1f)' % err_mr,
                'Tik. 0.1 (%.1f)' % err_ti_0_1,
                'Tik. 0.01 (%.1f)' % err_ti_1_0], loc=0)
    plt.xlabel('$E$ (eV)')
    plt.ylabel('$\phi(E)$')
    plt.savefig(img_directory+'maxent_vs_minnorm.pdf')
    
    plt.figure(7)
    err_cd = sp.mean(100*abs(phi_cd-phi_ref)/phi_ref)
    err_gd= sp.mean(100*abs(phi_gd-phi_ref)/phi_ref)
    err_cdgd = sp.mean(100*abs(phi_cdgd-phi_ref)/phi_ref)
    plt.loglog(x, yr, 'k', x, y11, 'r:', x, ycd, 'c--',x, ygd, 'b-.',x, ycdgd, 'g--')
    plt.legend(['reference', 
                'case 3 (%.1f)' % err11, 
                'case 3 + Cd  (%.1f)' % err_cd,
                'case 3 + Gd  (%.1f) ' % err_gd, 
                'case 3 + Cd + Gd  (%.1f)' % err_cdgd], loc=0)
    plt.xlabel('$E$ (eV)')
    plt.ylabel('$\phi(E)$')
    plt.savefig(img_directory+'filtered_unfold.pdf')
    
    plt.figure(8)
    plt.semilogy(x, yr, 'k', x, y11, 'r:', x, ycd, 'c--',x, ygd, 'b-.',x, ycdgd, 'g--')
    plt.legend(['reference', 
                'case 3 (%.1f)' % err11, 
                'case 3 + Cd  (%.1f)' % err_cd,
                'case 3 + Gd  (%.1f) ' % err_gd, 
                'case 3 + Cd + Gd  (%.1f)' % err_cdgd], loc=0)
    plt.xlabel('$E$ (eV)')
    plt.ylabel('$\phi(E)$')
    plt.axis([0.5,2,1e-4,1e1])
    plt.savefig(img_directory+'filtered_unfold_zoom.pdf')
    
