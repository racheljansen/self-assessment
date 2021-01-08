import random
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def loglik(theta,beta,eps0,alpha,len_beta,num_correct):
    
    """
    takes in a theta and beta value and uses a sigmoid function to compute likelihood of answering correctly
    1 / 1 + exp (-(theta-beta))
    also takes in a probability epsilon of making an incorrect inference
    
    """
    eps = eps0 - alpha * (num_correct/len_beta)
    
    assert eps >= 0, "Epsilon is a probability and must be between 0 and 1!"
    assert eps <= 1, "Epsilon is a probability and must be between 0 and 1!"


    return np.log( (1-eps) * (1 / (1 + np.exp(-(theta - beta)))) # 1-epsilon times prob right 
                  + eps * (1 / (1 + np.exp(theta - beta))) ) # plus epsilon times prob wrong

def loglikfail(theta,beta,eps0,alpha,len_beta,num_correct): # 1-loglik
    
    """
    takes in a theta and beta value and uses a sigmoid function to compute likelihood of answering incorrectly
    1 - (1 / 1 + exp (-(theta-beta)))
    """
    
    eps = eps0 - alpha * (num_correct/len_beta)
    
    assert eps >= 0, "Epsilon is a probability and must be between 0 and 1!"
    assert eps <= 1, "Epsilon is a probability and must be between 0 and 1!"
    
    return np.log( (1-eps) * (1 / (1 + np.exp(theta - beta)))
                 + eps * (1 / (1 + np.exp(-(theta - beta)))) )

def logprior(mu, sigma, x): # just gaussian distributions
    
    """
    gaussian distribution
    
    """
    
    return np.log((1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(x-mu)**2/(2*sigma**2)))

# determine number of passes in advance via the num_correct variable
def post_large_choice(theta, betas, mut, sigt, mub, sigb, eps0, alpha, len_beta, num_correct):

    '''
    inputs:
    theta: a value corresponding to a person's belief about their ability
    betas: a vector of values corresponding to the difficulty of each of a set of problems
        (could this come from performance in an inital pilot study?)
    mut, sigt: mean and standard deviation of theta
    mub, sigb: mean and standard deviation of betas
    num_correct: how many we deem the person got correct
    
    outputs:
    post: posterior
    
    '''
    assert len(betas) == len_beta, "Beta needs to have len_beta arguments in it!"
    
    assert num_correct <= len_beta , "Can't have more correct responses than total responses!"
    
    post = logprior(mut,sigt,theta) # start with prior only based on theta
    
    for beta in betas[:num_correct]:
        post += loglik(theta,beta,eps0,alpha,len_beta,num_correct) + logprior(mub,sigb,beta)
   
    for beta in betas[num_correct:]:
        post += loglikfail(theta,beta,eps0,alpha,len_beta,num_correct) + logprior(mub,sigb,beta)
            
    return post
    
    
# acceptance probability with choice
def acc_prob_choice(theta_old, betas_old, theta_new, betas_new, mut, sigt, mub, sigb, eps0, alpha, len_beta, num_correct):
    alpha = min(1, np.exp(post_large_choice(theta_new, betas_new, mut, sigt, mub, sigb, eps0, alpha, len_beta, num_correct)
                          - post_large_choice(theta_old, betas_old, mut, sigt, mub, sigb, eps0, alpha, len_beta, num_correct)))
    
    return alpha

# MCMC chain function
def MCMC(theta, betas, mut, sigt, mub, sigb, nsteps, len_beta, eps0, alpha, num_correct):        
    
    """
    input:
    theta: initial theta (should be random)
    betas: initial beta (vector)
    mut,sigt,mub,sigb: means and standard deviations
    nsteps: number of iterations of the chain
    len_beta: how long beta should be (meaning how many problems were solved)
    num_correct: out of the total number correct, how many were solved correctly
    
    Samples theta and then beta on each step: 
        in one iteration, calculate alpha, sample a theta, update alpha, then sample a beta

    output:
    sampled posterior thetas 
    sample posterior vector of beta values
    number of accepts of both thetas and betas
    """
    
    assert len(betas) == len_beta, "Beta needs to have len_beta arguments in it!"

    assert num_correct <= len_beta , "Can't have more correct responses than total responses!"
    
    naccept_theta = 0 # number of acceptances
    naccept_beta = 0 # number of acceptances

    #storage
    thetas_all = [] # fill with thetas
    betas_all = [] #np.empty((nsteps,len_beta)) # fill with vectors of betas

    for i in range(nsteps):

        u1 = np.random.uniform() # generate random uniform number u on [0,1] to compare to acceptance probability 
        theta_new = theta + np.random.normal(mut,sigt) # generate new values from the proposal distribution
        alpha1 = acc_prob_choice(theta, betas, theta_new, betas, mut, sigt, mub, sigb, eps0, alpha, len_beta, num_correct)

        # first sample a theta
        if u1 <= alpha1: # accept
            theta = theta_new
            naccept_theta += 1 # keep track of how many are accepted       

        else: # u > alpha (reject)
            pass
        
        u2 = np.random.uniform() # generate random uniform number u on [0,1] to compare to acceptance probability
        betas_new = betas + np.random.normal(mub,sigb,len_beta) #np.random.normal(mub,sigb,(1,len_beta)) #np.random.randn(len_beta,)  
        alpha2 = acc_prob_choice(theta, betas, theta, betas_new, mut, sigt, mub, sigb, eps0, alpha, len_beta, num_correct)
    
        # then sample a beta
        if u2 <= alpha2: # accept
            betas = betas_new
            naccept_beta += 1 # keep track of how many are accepted

        else: # u > alpha (reject)
            pass
       
        thetas_all.append(theta)
        betas_all.append(betas)

        
    return {'thetas': thetas_all, 
            'betas': betas_all,
            'naccept_theta': naccept_theta,
            'naccept_beta': naccept_beta,
           }
