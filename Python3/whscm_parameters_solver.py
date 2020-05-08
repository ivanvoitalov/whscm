import mpmath as mp
import numpy as np
import time
from scipy import integrate, special, optimize

def get_lam_c(a, R, beta1):
    """Function to compute the constant lam_c.

    Args:
        a (float): The a parameter of the WHSCM.
        R (float): The R parameter of the WHSCM.
        beta1 (float): The beta1 parameter of the WHSCM.

    Returns:
        lam_c (float): The switching point between the
                       two exponents of the double power-laws
                       in the WHSCM.

    """
    lam_c = (2. * a * np.exp(2.*R)) ** (1./(2. + beta1))
    return lam_c

def get_mu(lam, a, beta1, beta2, lam_c,
           verbose = 0):
    """Function to compute the latent parameter mu.

    Args:
        lam (float): The latent parameter lambda
                     of a node in the WHSCM.
        a (float): The a parameter of the WHSCM.
        beta1 (float): The beta1 parameter of the WHSCM.
        beta2 (float): The beta2 parameter of the WHSCM.
        lam_c (float): The switching point between the
                       two exponents of the double power-laws
                       in the WHSCM.

    Returns:
        mu (float): The latent parameter mu.
    """
    if lam >= 1 and lam <= lam_c:
        mu = a * lam**(-beta1)
    elif lam > lam_c:
        mu = a * (lam_c**(beta2 - beta1)) * lam**(-beta2)
    else:
        if verbose == 1:
            print("ERROR (get_mu): Lambda parameter has to be between 1 and infinity.")
        mu = None
    return mu

def rho_lam(lam, alpha1, alpha2, lam_c,
            verbose = 0):
    """Function to compute the PDF of the lambda parameter.

    Args:
        lam (float): The latent parameter lambda
                     of a node in the WHSCM.
        alpha1 (float): The alpha1 parameter of the WHSCM.
        alpha2 (float): The alpha2 parameter of the WHSCM.
        lam_c (float): The switching point between the
                       two exponents of the double power-laws
                       in the WHSCM.

    Returns:
        rho (float): the value of the lambda's PDF.
    """
    A1 = (alpha1 - 1.) * (alpha2 - 1.) / ( (lam_c**(1.-alpha1))*(alpha1 - alpha2) + alpha2 - 1. )
    A2 = A1 * (lam_c**(alpha2 - alpha1))

    if lam >= 1 and lam <= lam_c:
        rho = A1 * lam**(-alpha1)
    elif lam > lam_c:
        rho = A2 * lam**(-alpha2)
    else:
        if verbose == 1:
            print("ERROR (rho_lam): Lambda parameter has to be between 1 and infinity.")
        rho = None
    return rho

def get_A1_const(alpha1, alpha2, lam_c):
    """Function to compute the constant A1.

    Args:
        alpha1 (float): The alpha1 parameter of the WHSCM.
        alpha2 (float): The alpha2 parameter of the WHSCM.
        lam_c (float): The switching point between the
                       two exponents of the double power-laws
                       in the WHSCM.

    Returns:
        A1 (float): The A1 constant of the WHSCM.
    """
    A1 = (alpha1 - 1.) * (alpha2 - 1.) / ( (lam_c**(1.-alpha1))*(alpha1 - alpha2) + alpha2 - 1. )
    return A1

def get_A2_const(alpha1, alpha2, lam_c, A1):
    """Function to compute the constant A2.

    Args:
        alpha1 (float): The alpha1 parameter of the WHSCM.
        alpha2 (float): The alpha2 parameter of the WHSCM.
        lam_c (float): The switching point between the
                       two exponents of the double power-laws
                       in the WHSCM.
        A1 (float): The A1 constant of the WHSCM.

    Returns:
        A2 (float): The A2 constant of the WHSCM.
    """
    A2 = A1 * (lam_c**(alpha2 - alpha1))
    return A2

def get_kbar_lam_approx(lam, n, R, a, alpha1, alpha2, beta1, beta2,
                        verbose = 0):
    """Function to compute the expected degree as a function of
       the latent parameter lam approximately. See Appendix of
       the WHSCM paper for details of approximation.

    Args:
        lam (float): The latent parameter lambda of a node.
        n (int): The number of nodes in a graph.
        R (float): The R parameter of the WHSCM.
        a (float): The a parameter of the WHSCM.
        alpha1 (float): The alpha1 parameter of the WHSCM.
        alpha2 (float): The alpha2 parameter of the WHSCM.
        beta1 (float): The beta1 parameter of the WHSCM.
        beta2 (float): The beta2 parameter of the WHSCM.

    Returns:
        kbar_lam_approx (float): The expected degree of a node
                                 with the latent parameter lambda
                                 in the WHSCM.
    """
    lam_c = get_lam_c(a, R, beta1)
    A1 = get_A1_const(alpha1, alpha2, lam_c)
    A2 = get_A2_const(alpha1, alpha2, lam_c, A1)

    if lam >= 1 and lam <= lam_c:
        term1 = (n*A1*(lam**(1.+beta1)))/(a * (alpha1-2.) * np.exp(2*R))
        term2 = special.hyp2f1(1., (alpha1-2.)/beta1, 1. + (alpha1-2.)/beta1, - (lam**beta1))
        term3 = (lam_c**(2.-alpha1)) * special.hyp2f1(1., (alpha1-2.)/beta1, 1. + (alpha1-2.)/beta1, - ((lam/lam_c)**beta1))
        term4 = (n * A2 * (lam_c)**(1.-alpha2)) / (alpha2 - 1.)
        term5 = special.hyp2f1(1., alpha2 - 1., alpha2, -(a * np.exp(2*R))/(lam_c * lam**(1.+beta1)))
        kbar_lam_approx = term1 * (term2 - term3) + term4*term5

    elif lam > lam_c:
        term1 = n * A1 / (alpha1-1.)
        term2 = special.hyp2f1(1., (alpha1-1.)/(1.+beta1), (alpha1+beta1)/(1.+beta1), -(a*np.exp(2.*R))/lam)
        term3 = ((lam_c)**(1.-alpha1)) * special.hyp2f1(1., (alpha1-1.)/(1+beta1), (alpha1+beta1)/(1.+beta1), -(a*np.exp(2.*R))/(lam*(lam_c)**(1.+beta1)))
        term4 = (n * A2 * (lam_c)**(1.-alpha2)) / (alpha2 - 1.)
        kbar_lam_approx = term1 * (term2 - term3) + term4

    else:
        if verbose == 1:
            print("ERROR (get_kbar_lam_approx): Lambda parameter "+\
                  "has to be between 1 and infinity.")
        kbar_lam_approx = None

    return kbar_lam_approx

def get_sbar_lam_approx(lam, n, R, a, alpha1, alpha2, beta1, beta2,
                        verbose = 0):
    """Function to compute the expected strength as a function of
       the latent parameter lam approximately. See Appendix of
       the WHSCM paper for details of approximation.

    Args:
        lam (float): The latent parameter lambda of a node.
        n (int): The number of nodes in a graph.
        R (float): The R parameter of the WHSCM.
        a (float): The a parameter of the WHSCM.
        alpha1 (float): The alpha1 parameter of the WHSCM.
        alpha2 (float): The alpha2 parameter of the WHSCM.
        beta1 (float): The beta1 parameter of the WHSCM.
        beta2 (float): The beta2 parameter of the WHSCM.

    Returns:
        sbar_lam_approx (float): The expected strength of a node
                                 with the latent parameter lambda
                                 in the WHSCM.
    """
    lam_c = get_lam_c(a, R, beta1)
    A1 = get_A1_const(alpha1, alpha2, lam_c)
    A2 = get_A2_const(alpha1, alpha2, lam_c, A1)

    if lam >= 1 and lam <= lam_c:
        term1 = (n * A1 * lam**(1.+beta1)) / (np.exp(2*R) * (lam_c)**(alpha1) * a * a * beta1 * (1. + (lam_c / lam)**beta1))
        term2 = -(lam_c**(2.+beta1))
        term3 = ((lam_c**(alpha1))/(1. + lam**beta1)) * (lam**beta1 + lam_c**beta1)
        term4 = (2.+beta1-alpha1)*(lam**beta1 + lam_c**beta1)/(alpha1-2.)
        term5 = (lam_c**alpha1) * special.hyp2f1(1., (alpha1-2.)/beta1, 1. + (alpha1-2.)/beta1, - (lam**beta1))
        term6 = (lam_c**2.) * special.hyp2f1(1., (alpha1-2.)/beta1, 1. + (alpha1-2.)/beta1, - ((lam/lam_c)**beta1))
        term7 = (n * A2 * lam**(beta1) * (lam_c)**(1.-alpha2)) / a
        term8 = (mp.gamma(alpha2-1.)/mp.gamma(alpha2)) * special.hyp2f1(1., alpha2-1., alpha2, - (a*np.exp(2.*R))/(lam**(1.+beta1) * lam_c))
        sbar_lam_approx = term1*(term2 + term3 + term4*(term5 - term6)) + term7*term8

    elif lam > lam_c:
        term1 = (n * A1 * mp.gamma((-beta1+alpha1-1.)/(1.+beta1))) / (a * (lam_c**alpha1) * (1.+beta1) * mp.gamma(alpha1/(1.+beta1)))
        term2 = (lam_c**alpha1) * special.hyp2f1(1., (alpha1/(1.+beta1)) - 1., alpha1/(1.+beta1), - (a*np.exp(2.*R))/lam)
        term3 = (lam_c**(1.+beta1)) * special.hyp2f1(1., (alpha1/(1.+beta1)) - 1., alpha1/(1.+beta1), - (a*np.exp(2.*R))/(lam * lam_c**(1.+beta1)))
        term4 = (n * A2 * (lam**beta2) * (lam_c**(1.+beta1-alpha2-beta2))) / (a*(alpha2-1.))
        term5 = special.hyp2f1(1., (alpha2-1.)/beta2, 1. + (alpha2-1.)/beta2, -(lam/(lam_c))**beta2)
        sbar_lam_approx = term1*(term2 - term3) + term4*term5

    else:
        if verbose == 1:
            print("ERROR (get_sbar_lam_approx): Lambda parameter "+\
                  "has to be between 1 and infinity.")
        sbar_lam_approx = None

    return sbar_lam_approx

def get_avek_approx(n, R, a, alpha1, alpha2, beta1, beta2):
    """Function to compute average degree of the WHSCM network
       using the approximated expression for the $\bar{k}(\lambda)$.

       Args:
           n (int): The number of nodes in a graph.
           R (float): The R parameter of the WHSCM.
           a (float): The a parameter of the WHSCM.
           alpha1 (float): The alpha1 parameter of the WHSCM.
           alpha2 (float): The alpha2 parameter of the WHSCM.
           beta1 (float): The beta1 parameter of the WHSCM.
           beta2 (float): The beta2 parameter of the WHSCM.

       Returns:
           avek (float): The average degree of the WHSCM network.
    """
    lam_c = get_lam_c(a, R, beta1)
    A1 = get_A1_const(alpha1, alpha2, lam_c)
    A2 = get_A2_const(alpha1, alpha2, lam_c, A1)
    func1 = lambda lam, n, R, a, alpha1, alpha2, beta1, beta2: get_kbar_lam_approx(lam, n, R, a, alpha1, alpha2, beta1, beta2) * A1 * lam**(-alpha1)
    func2 = lambda lam, n, R, a, alpha1, alpha2, beta1, beta2: get_kbar_lam_approx(lam, n, R, a, alpha1, alpha2, beta1, beta2) * A2 * lam**(-alpha2)
    term1 = integrate.quad(func1, 1.0, lam_c, args = (n, R, a, alpha1, alpha2, beta1, beta2))
    term2 = integrate.quad(func2, lam_c, np.inf, args = (n, R, a, alpha1, alpha2, beta1, beta2))
    kbar = term1[0] + term2[0]
    return kbar

def conn_prob(lam, lam_prime, a, R, beta1, beta2, lam_c,
              verbose = 0):
    """Function to compute the connection probability
       between the nodes with the parameters lam and lam_prime.

    Args:
        lam (float): the latent parameter of the first node.
        lam_prime (float): the latent parameter of the second node.
        a (float): The a parameter of the WHSCM.
        R (float): The R parameter of the WHSCM.
        beta1 (float): The beta1 parameter of the WHSCM.
        beta2 (float): The beta2 parameter of the WHSCM.
        lam_c (float): The switching point between the
                       two exponents of the double power-laws
                       in the WHSCM.

    Returns:
        p (float): the connection probability.
    """
    if lam >= 1 and lam <= lam_c:
        if lam_prime >= 1 and lam_prime <= lam_c:
            p = 1.0 / (1.0 + a*np.exp(2.*R) * (lam**(-beta1) + lam_prime**(-beta1)) / (lam * lam_prime))
        elif lam_prime > lam_c:
            p = 1.0 / (1.0 + a*np.exp(2.*R) * (lam**(-beta1) + (lam_c**(beta2-beta1)) * lam_prime**(-beta2)) / (lam * lam_prime))
        else:
            if verbose == 1:
                print("ERROR (conn_prob): Lambda parameter (lam_prime) has to be between 1 and infinity.")
            p = None

    elif lam > lam_c:
        if lam_prime >= 1 and lam_prime <= lam_c:
            p = 1.0/(1.0 + a*np.exp(2.*R)* ((lam_c**(beta2-beta1)) * (lam**(-beta2)) + lam_prime**(-beta1)) / (lam * lam_prime))
        elif lam_prime > lam_c:
            p = 1.0/(1.0 + a*np.exp(2.*R)* ((lam_c**(beta2-beta1)) * (lam**(-beta2)) + (lam_c**(beta2-beta1)) * (lam_prime**(-beta2))) / (lam * lam_prime))
        else:
            if verbose == 1:
                print("ERROR (conn_prob): Lambda parameter (lam_prime) has to be between 1 and infinity.")
            p = None

    else:
        if verbose == 1:
            print("ERROR (conn_prob): Lambda parameter (lam) has to be between 1 and infinity.")
        p = None

    return p

def integrand_kbar(lam_prime, lam, a, R, alpha1, alpha2, beta1, beta2):
    """Function to compute the integrand used in the 
       double integral for accurate computation of the
       average degree in the WHSCM.

        Args:
            lam_prime (float): The latent parameter of the second node.
            lam (float): The latent parameter of the first node.
            a (float): The a parameter of the WHSCM.
            R (float): The R parameter of the WHSCM.
            alpha1 (float): The alpha1 parameter of the WHSCM.
            alpha2 (float): The alpha2 parameter of the WHSCM.
            beta1 (float): The beta1 parameter of the WHSCM.
            beta2 (float): The beta2 parameter of the WHSCM.
        
        Returns:
            integrand (float): The resulting integrand.
    """
    lam_c = (2.*a*np.exp(2.*R))**(1./(2. + beta1))
    p = conn_prob(lam, lam_prime, a, R, beta1, beta2, lam_c)
    rho1 = rho_lam(lam_prime, alpha1, alpha2, lam_c)
    rho2 = rho_lam(lam, alpha1, alpha2, lam_c)
    integrand = p*rho1*rho2
    return integrand

def integrand_expected_k_lam(lam_prime, lam, n, R, a, alpha1, alpha2, beta1, beta2):
    """Function to compute the integrand used in the
       integral for accurate computation of the
       expected degree of a node with the latent
       parameter lam.

    Args:
        lam_prime (float): The latent parameter of the second node.
        lam (float): The latent parameter of the first node.
        n (int): The number of nodes in the network.
        R (float): The R parameter of the WHSCM.
        a (float): The a parameter of the WHSCM.
        alpha1 (float): The alpha1 parameter of the WHSCM.
        alpha2 (float): The alpha2 parameter of the WHSCM.
        beta1 (float): The beta1 parameter of the WHSCM.
        beta2 (float): The beta2 parameter of the WHSCM.

    Returns:
        integrand (float): The resulting integrand.
    """
    lam_c = (2.*a*np.exp(2.*R))**(1./(2. + beta1))
    rho1 = rho_lam(lam_prime, alpha1, alpha2, lam_c)
    p = conn_prob(lam, lam_prime, a, R, beta1, beta2, lam_c)
    integrand = rho1*p
    return integrand

def integrand_expected_s_lam(lam_prime, lam, n, R, a, alpha1, alpha2, beta1, beta2):
    """Function to compute the integrand used in the
       integral for accurate computation of the
       expected strength of a node with the latent
       parameter lam.

    Args:
        lam_prime (float): The latent parameter of the second node.
        lam (float): The latent parameter of the first node.
        n (int): The number of nodes in the network.
        R (float): The R parameter of the WHSCM.
        a (float): The a parameter of the WHSCM.
        alpha1 (float): The alpha1 parameter of the WHSCM.
        alpha2 (float): The alpha2 parameter of the WHSCM.
        beta1 (float): The beta1 parameter of the WHSCM.
        beta2 (float): The beta2 parameter of the WHSCM.

    Returns:
        integrand (float): The resulting integrand.
    """
    lam_c = (2.*a*np.exp(2.*R))**(1./(2. + beta1))
    rho1 = rho_lam(lam_prime, alpha1, alpha2, lam_c)
    p = conn_prob(lam, lam_prime, a, R, beta1, beta2, lam_c)
    integrand = rho1*p
    integrand = integrand * (1. / (get_mu(lam, a, beta1, beta2, lam_c) + get_mu(lam_prime, a, beta1, beta2, lam_c)))
    return integrand

def expected_k_lam(lam, n, R, a,
                   alpha1, alpha2, beta1, beta2):
    """Function to compute the expected
       degree as a function of the lambda
       parameter. Integration is performed
       numerically and is expected to give
       an "exact" answer.
       Covers the eta > 1 case.
        
        Args:
            lam (float): The latent parameter of the first node.
            n (int): The number of nodes in the network.
            R (float): The R parameter of the WHSCM.
            a (float): The a parameter of the WHSCM.
            alpha1 (float): The alpha1 parameter of the WHSCM.
            alpha2 (float): The alpha2 parameter of the WHSCM.
            beta1 (float): The beta1 parameter of the WHSCM.
            beta2 (float): The beta2 parameter of the WHSCM.

        Returns:
            exp_k_lambda (float): The expected degree of a node
                                  with a latent parameter lam.
    """
    ans, err = integrate.quad(integrand_expected_k_lam, 1.0, np.inf,
                              args = (lam, n, R, a, alpha1, alpha2, beta1, beta2),
                              limit = 1000)
    exp_k_lambda = n * ans
    return exp_k_lambda

def expected_s_lam(lam, n, R, a,
                   alpha1, alpha2, beta1, beta2):
    """Function to compute the expected
       strength as a function of the lambda
       parameter. Integration is performed
       numerically and is expected to give
       an "exact" answer.
       Covers the eta > 1 case.
        
        Args:
            lam (float): The latent parameter of the first node.
            n (int): The number of nodes in the network.
            R (float): The R parameter of the WHSCM.
            a (float): The a parameter of the WHSCM.
            alpha1 (float): The alpha1 parameter of the WHSCM.
            alpha2 (float): The alpha2 parameter of the WHSCM.
            beta1 (float): The beta1 parameter of the WHSCM.
            beta2 (float): The beta2 parameter of the WHSCM.

        Returns:
            exp_s_lambda (float): The expected strength of a node
                                  with a latent parameter lam.
    """
    ans, err = integrate.quad(integrand_expected_s_lam, 1.0, np.inf,
                              args = (lam, n, R, a, alpha1, alpha2, beta1, beta2),
                              limit = 1000)
    exp_s_lambda = n * ans
    return exp_s_lambda

def get_avek(n, R, a, alpha1, alpha2, beta1, beta2):
    """Function to compute the average degree
    in the WHSCM network in the case eta > 1.

    Args:
        n (int): The number of nodes in the network.
        R (float): The R parameter of the WHSCM.
        a (float): The a parameter of the WHSCM.
        alpha1 (float): The alpha1 parameter of the WHSCM.
        alpha2 (float): The alpha2 parameter of the WHSCM.
        beta1 (float): The beta1 parameter of the WHSCM.
        beta2 (float): The beta2 parameter of the WHSCM.

    Returns:
        kabr (float): The average degree in the WHSCM network.
    """
    options = {'limit': 1000}
    ans, err = integrate.nquad(integrand_kbar, [[1., np.inf], [1.,  np.inf]],
                               args = (a, R, alpha1, alpha2, beta1, beta2),
                               opts=[options, options])
    kbar = n*ans
    return kbar

def get_solution_approx(n, kbar_target, sigma0_target,
                        gamma_target, eta_target,
                        verbose = 0):
    """Function to compute the solutions for the (R, a)
       parameters of the WHSCM using the approximated
       versions of expected degrees and strengths
       as functions of lambda. This function is used
       for getting a rough estimate of the (R, a) that
       are then used as an initial guess in a more
       precise routine. If eta = 1.0, then the exact solution
       is obtained directly.

       Args:
            n (int): The network size.
            kbar_target (float): The target average degree
                                 of the WHSCM network.
            sigma0_target (float):  The target sigma0 parameter
                                    of the WHSCM network.
            gamma_target (float): The target degree distribution
                                  power-law exponent (> 2).
            eta_target (float): The target strength-degree scaling
                                exponent (>= 1).
            R_guess (float, optional): The initial guess for the
                                       R parameter solution.
            a_guess (float, optional): The initial guess for the
                                       a parameter solution. Ignored
                                       if eta = 1.0.
            lam0 (float, optional): The lambda0 constant with respect
                                    to which the sigma0 equation is
                                    solved.

        Returns:
            R (float): The approximate solution for the R parameter.
            a (float): The approximate solution for the a parameter.
    """
    if eta_target > 1.0:
        alpha1 = 1. + eta_target*(gamma_target - 1.)
        beta1 = (gamma_target - (gamma_target - 2.)/gamma_target)*(eta_target - 1.)
        alpha2 = 1. + ((alpha1 - 1.) / (1. + beta1)) * (1. + (gamma_target - 2.)*(1. - 1./eta_target))
        beta2 = alpha2 - 1. + (eta_target*(alpha2 - 1.))/(gamma_target - 1.)

        R_guess, a_guess = get_solution_approx(n, kbar_target, sigma0_target,
                                               gamma_target, 1.0)
        if verbose == 1:
            print("Initial guess (from eta = 1):", R_guess, a_guess)
        if R_guess == None:
            R_guess = 10.1
        if a_guess == None:
            a_guess = 1.1

        def get_lam0_kbar(x, target_value):
            def f(lam):
                kbar_lam = get_kbar_lam_approx(lam,n,x[0],x[1],alpha1,alpha2,beta1,beta2)
                if kbar_lam == None or np.isnan(kbar_lam):
                    return np.inf
                else:
                    return (kbar_lam - target_value)**2
            sol = optimize.fsolve(f, 1.1)
            return sol

        def func(x):
            if (x[0] < 1e-16) or (np.isnan(x[0])) or (x[1] < 1e-16) or (np.isnan(x[1])):
                if verbose == 1:
                    print("Current R = %.8f, current a = %.8f, current residue = inf." % (x[0], x[1]))
                return [np.inf, 0]
            
            # find lambda that correspond to approximately 1.1 * kbar value of
            # the expected degree \bar{k}(lambda)
            lam_test = get_lam0_kbar(x, 1.1*kbar_target)

            if lam_test <= 1.0 or lam_test == None:
                if verbose == 1:
                    print("Current R = %.8f, current a = %.8f, current residue = inf." % (x[0], x[1]))
                return [np.inf, 0]
            else:
                kbar_estimate = get_avek_approx(n,x[0],x[1],alpha1,alpha2,beta1,beta2)
                sbar_lam_estimate = get_sbar_lam_approx(lam_test,n,x[0],x[1],alpha1,alpha2,beta1,beta2)
                kbar_lam_estimate = get_kbar_lam_approx(lam_test,n,x[0],x[1],alpha1,alpha2,beta1,beta2)
                if (kbar_estimate == None) or (kbar_lam_estimate == None) or (sbar_lam_estimate == None):
                    if verbose == 1:
                        print("Current R = %.8f, current a = %.8f, current residue = inf." % (x[0], x[1]))
                    return [np.inf, 0]
                else:
                    r = ((kbar_estimate - kbar_target)/kbar_target)**2 + ((sbar_lam_estimate / (kbar_lam_estimate)**eta_target - sigma0_target )/sigma0_target)**2
                    if verbose == 1:
                        print("Current R = %.8f, current a = %.8f, current residue = %.12f" % (x[0], x[1], r))
                    return [r, 0]
        
        print("Starting the approximate solver...")
        sol = optimize.fsolve(func, x0 = [R_guess, a_guess], xtol = 1e-04)
        R, a = sol
        lam_test = get_lam0_kbar([R,a], 1.1*kbar_target)
        if lam_test <= 1.0 or lam_test == None:
            current_r = np.inf
        else:
            kbar_estimate = get_avek_approx(n,R,a,alpha1,alpha2,beta1,beta2)
            sbar_lam_estimate = get_sbar_lam_approx(lam_test,n,R,a,alpha1,alpha2,beta1,beta2)
            kbar_lam_estimate = get_kbar_lam_approx(lam_test,n,R,a,alpha1,alpha2,beta1,beta2)
            if (kbar_estimate == None) or (kbar_lam_estimate == None) or (sbar_lam_estimate == None):
                current_r = np.inf
            else:
                current_r = ((kbar_estimate - kbar_target)/kbar_target)**2 + ((sbar_lam_estimate / (kbar_lam_estimate)**eta_target - sigma0_target )/sigma0_target)**2
            
        while current_r > 0.001:
            R, a = np.random.uniform(0.9,1.1)*R, np.random.uniform(0.9,1.1)*a
            if verbose == 1:
                print("Residue remained too large, restarting with perturbed initial guess: R = %.8f, a = %.8f" % (R, a))
            sol = optimize.fsolve(func, x0 = [R, a], xtol = 1e-04)
            R, a = sol
            lam_test = get_lam0_kbar([R,a], 1.1*kbar_target)
            if lam_test <= 1.0 or lam_test == None:
                current_r = np.inf
            else:
                kbar_estimate = get_avek_approx(n,R,a,alpha1,alpha2,beta1,beta2)
                sbar_lam_estimate = get_sbar_lam_approx(lam_test,n,R,a,alpha1,alpha2,beta1,beta2)
                kbar_lam_estimate = get_kbar_lam_approx(lam_test,n,R,a,alpha1,alpha2,beta1,beta2)
                if (kbar_estimate == None) or (kbar_lam_estimate == None) or (sbar_lam_estimate == None):
                    current_r = np.inf
                else:
                    current_r = ((kbar_estimate - kbar_target)/kbar_target)**2 + ((sbar_lam_estimate / (kbar_lam_estimate)**eta_target - sigma0_target )/sigma0_target)**2
        
        # print out approximate solver estimates for the average degree
        # and sigma0 with the (R, a) solution found
        result_kbar = get_avek_approx(n,R,a,alpha1,alpha2,beta1,beta2)
        result_sigma0 = get_sbar_lam_approx(lam_test,n,R,a,alpha1,alpha2,beta1,beta2) / (get_kbar_lam_approx(lam_test,n,R,a,alpha1,alpha2,beta1,beta2))**(eta_target)
        print("==========")
        print("Approximate solver solution: R = %.12f, a = %.12f" % (R, a))
        print("Estimated average degree with the current choice of (R, a), approximate solver:", float(result_kbar))
        print("Estimated sigma0 with the current choice of (R, a), approximate solver:", float(result_sigma0))
        print("==========")

    elif eta_target == 1.0:
        a = 1. / (2. * sigma0_target)
        integrand_eta1 = lambda x, gamma, R_sol, a_sol: special.hyp2f1(1.0, gamma - 1.0, gamma, -(2.*a_sol*np.exp(2.*R_sol))/x) * x**(-gamma)
        kbar_func = lambda n, gamma, R_sol, a_sol: n*(gamma-1.)*integrate.quad(integrand_eta1, 1.0, np.inf, args = (gamma, R_sol, a_sol))[0]
        
        # Lerch Phi closed form expression, seems to be unstable for some combinations of input parameters
        #kbar_func = lambda n, gamma, eta, R_sol, a: (n * (gamma - 1.)**2) * mp.re(mp.lerchphi(-2.*a*mp.exp(2.*R_sol), 2.0, gamma - 1.))
        
        # terminate if kbar is too large to be realized in the
        # HSCM case
        if kbar_target > kbar_func(n, gamma_target, 0.0, a):
            
            print("ERROR (get_solution_approx): "+\
                  "the requested target kbar is larger than the maximum "+\
                  "possible kbar in the eta = 1 case. "+\
                  "Try rescaling strengths (sigma0 parameter).")
            return None, None

        # use bisection method to find the R root
        def bisection(R_guess_min, R_guess_max, eps = 10e-6):
            kbar_guess_min = kbar_func(n, gamma_target, R_guess_min, a)
            kbar_guess_max = kbar_func(n, gamma_target, R_guess_max, a)

            #if either of the endpoints is nan, reduce/increase it
            while np.isnan(float(kbar_guess_max)):
                R_guess_max = 0.95*R_guess_max
                kbar_guess_max = kbar_func(n, gamma_target, R_guess_max, a)
            while np.isnan(float(kbar_guess_min)):
                R_guess_min = 1.05*R_guess_min
                kbar_guess_min = kbar_func(n, gamma_target, R_guess_min, a)

            growing = False
            if kbar_guess_max > kbar_guess_min:
                growing = True
            else:
                growing = False
            mid_point = (R_guess_min + R_guess_max) / 2.0
            kbar_guess = kbar_func(n, gamma_target, mid_point, a)
            current_error = ((kbar_guess - kbar_target) / float(kbar_target))**2
            if growing:
                while current_error > eps:
                    if kbar_guess < kbar_target:
                        # search to the right
                        R_guess_min = mid_point
                    else:
                        # search to the left
                        R_guess_max = mid_point
                    mid_point = (R_guess_min + R_guess_max) / 2.0
                    kbar_guess = kbar_func(n, gamma_target, mid_point, a)
                    current_error = ((kbar_guess - kbar_target) / float(kbar_target))**2
            else:
                while current_error > eps:
                    if kbar_guess < kbar_target:
                        # search to the left
                        R_guess_max = mid_point
                    else:
                        # search to the right
                        R_guess_min = mid_point
                    mid_point = (R_guess_min + R_guess_max) / 2.0
                    kbar_guess = kbar_func(n, gamma_target, mid_point, a)
                    current_error = ((kbar_guess - kbar_target) / float(kbar_target))**2
            return mid_point
        
        R = bisection(10e-12, 100, eps = 10e-6)

        result_kbar = kbar_func(n, gamma_target, R, a)
        result_sigma0 = 1./(2.*a)
        print("==========")
        print("Approximate solver solution: R = %.12f, a = %.12f" % (R, a))
        print("Estimated average degree with the current choice of (R, a), approximate solver:", float(result_kbar))
        print("Estimated sigma0 with the current choice of (R, a), approximate solver:", float(result_sigma0))
        print("==========")
        
    else:
        if verbose == 1:
            print("ERROR (get_solution_approx): "+\
                  "eta_target parameter has to be greater or equal to 1.")
        R, a = None, None

    return R, a

def get_solution(n, kbar_target, sigma0_target,
                 gamma_target, eta_target,
                 verbose = 0):
    """Function to compute the solutions for the (R, a)
       parameters of the WHSCM. For the eta > 1 case,
       the approximate solutions are obtained first,
       and then used as an initial guess for this function.

       Args:
            n (int): The network size.
            kbar_target (float): The target average degree
                                 of the WHSCM network.
            sigma0_target (float):  The target sigma0 parameter
                                    of the WHSCM network.
            gamma_target (float): The target degree distribution
                                  power-law exponent (> 2).
            eta_target (float): The target strength-degree scaling
                                exponent (>= 1).
            R_guess (float, optional): The initial guess for the
                                       R parameter solution.
            a_guess (float, optional): The initial guess for the
                                       a parameter solution. Ignored
                                       if eta = 1.0.
            lam0 (float, optional): The lambda0 constant with respect
                                    to which the sigma0 equation is
                                    solved.

        Returns:
            R (float): The exact solution for the R parameter.
            a (float): The exact solution for the a parameter.
    """
    if eta_target > 1.0:

        R_init, a_init = get_solution_approx(n, kbar_target, sigma0_target,
                                             gamma_target, eta_target,
                                             verbose = verbose)

        alpha1 = 1. + eta_target*(gamma_target - 1.)
        beta1 = (gamma_target - (gamma_target - 2.)/gamma_target)*(eta_target - 1.)
        alpha2 = 1. + ((alpha1 - 1.) / (1. + beta1)) * (1. + (gamma_target - 2.)*(1. - 1./eta_target))
        beta2 = alpha2 - 1. + (eta_target*(alpha2 - 1.))/(gamma_target - 1.)

        def get_lam0_kbar(x, target_value):
            def f(lam):
                kbar_lam = expected_k_lam(lam,n,x[0],x[1],alpha1,alpha2,beta1,beta2)
                if kbar_lam == None or np.isnan(kbar_lam):
                    return np.inf
                else:
                    return (kbar_lam - target_value)**2
            sol = optimize.fsolve(f, 1.1)
            return sol

        def func(x):
            if (x[0] < 1e-16) or (np.isnan(x[0])) or (x[1] < 1e-16) or (np.isnan(x[1])):
                if verbose == 1:
                    print("Current R = %.8f, current a = %.8f, current residue = inf." % (x[0], x[1]))
                return [np.inf, 0]
            
            # find lambda that correspond to approximately 1.1 * kbar value of
            # the expected degree \bar{k}(lambda)
            lam_test = get_lam0_kbar(x, 1.1*kbar_target)

            if lam_test <= 1.0 or lam_test == None:
                if verbose == 1:
                    print("Current R = %.8f, current a = %.8f, current residue = inf." % (x[0], x[1]))
                return [np.inf, 0]
            else:
                kbar_estimate = get_avek(n,x[0],x[1],alpha1,alpha2,beta1,beta2)
                sbar_lam_estimate = expected_s_lam(lam_test,n,x[0],x[1],alpha1,alpha2,beta1,beta2)
                kbar_lam_estimate = expected_k_lam(lam_test,n,x[0],x[1],alpha1,alpha2,beta1,beta2)
                if (kbar_estimate == None) or (kbar_lam_estimate == None) or (sbar_lam_estimate == None):
                    if verbose == 1:
                        print("Current R = %.8f, current a = %.8f, current residue = inf." % (x[0], x[1]))
                    return [np.inf, 0]
                else:
                    r = ((kbar_estimate - kbar_target)/kbar_target)**2 + ((sbar_lam_estimate / (kbar_lam_estimate)**eta_target - sigma0_target )/sigma0_target)**2
                    if verbose == 1:
                        print("Current R = %.8f, current a = %.8f, current residue = %.12f" % (x[0], x[1], r))
                    return [r, 0]

        
        print("Starting the exact solver...")
        sol = optimize.fsolve(func, x0 = [R_init, a_init], xtol = 1e-04)
        R, a = sol
        lam_test = get_lam0_kbar([R,a], 1.1*kbar_target)
        kbar_estimate = get_avek(n,R,a,alpha1,alpha2,beta1,beta2)
        sbar_lam_estimate = expected_s_lam(lam_test,n,R,a,alpha1,alpha2,beta1,beta2)
        kbar_lam_estimate = expected_k_lam(lam_test,n,R,a,alpha1,alpha2,beta1,beta2)
        current_r = ((kbar_estimate - kbar_target)/kbar_target)**2 + ((sbar_lam_estimate / (kbar_lam_estimate)**eta_target - sigma0_target )/sigma0_target)**2
        
        # we do not perform restarting of the solver with perturbed initial guess here as the
        # precise solver is too slow, the approximate solver should obtain close enough solution alredy
        
        # print out exact solver estimates for the average degree
        # and sigma0 with the (R, a) solution found
        result_kbar = get_avek(n,R,a,alpha1,alpha2,beta1,beta2)
        result_sigma0 = expected_s_lam(lam_test,n,R,a,alpha1,alpha2,beta1,beta2) / (expected_k_lam(lam_test,n,R,a,alpha1,alpha2,beta1,beta2))**(eta_target)
        print("==========")
        print("Exact solver solution: R = %.12f, a = %.12f" % (R, a))
        print("Estimated average degree with the current choice of (R, a):", result_kbar)
        print("Estimated sigma0 with the current choice of (R, a):", result_sigma0)
        print("==========")

    elif eta_target == 1.0:
        # the exact numerical solution is obtained in the
        # get_solution_approx(...) function
        R, a = get_solution_approx(n, kbar_target, sigma0_target,
                                   gamma_target, eta_target)
    else:
        print("ERROR (get_solution_approx): "+\
              "eta_target parameter has to be greater or equal to 1.")
        R, a = None, None

    return R, a
