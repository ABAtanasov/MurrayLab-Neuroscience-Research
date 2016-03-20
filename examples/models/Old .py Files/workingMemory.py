"""
Perceptual decision-making task, loosely based on the random dot motion
discrimination task.

  Response of neurons in the lateral intraparietal area during a combined visual
  discrimination reaction time task.
  J. D. Roitman & M. N. Shadlen, JNS 2002.

  http://www.jneurosci.org/content/22/21/9475.abstract

"""
#You have to change hi/lo to yes/no

from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin  = 2
N    = 100
Nout = 2

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

# Input labels
POS = 0
NEG = 1

yes = 1
no = 0.2

#-----------------------------------------------------------------------------------------
# Recurrent connectivity
#-----------------------------------------------------------------------------------------

Crec = tasktools.generate_Crec(ei, p_exc=0.1, p_inh=0.5, seed=1066)


#-----------------------------------------------------------------------------------------
# Output connectivity
#-----------------------------------------------------------------------------------------

Cout = np.zeros((Nout, N))
Cout[:,EXC] = 1

#-----------------------------------------------------------------------------------------
# Task structure
#-----------------------------------------------------------------------------------------

conditions = [(yes, yes), (yes, no), (no, yes), (no, no)]
in_outs = ['equal', 'notequal']
nconditions = len(conditions)#*len(in_outs)
pcatch      = 0     #implement catch trials

def generate_trial(rng, dt, params):
    #-------------------------------------------------------------------------------------
    # Select task condition
    #-------------------------------------------------------------------------------------

    catch_trial = False
    if params['name'] in ['gradient', 'test']:
        if params.get('catch', rng.rand() < pcatch): #pcatch chance of a catch trial
            catch_trial = True
        else:
            cond    = params.get('conditions', conditions[rng.choice(len(conditions))])
            #Revert back to cohs if this brings up an issue
            in_out = params.get('in_out', rng.choice(in_outs)) 
    elif params['name'] == 'validation':
        b = params['minibatch_index'] % (nconditions + 1)
        if b == 0:
            catch_trial = True
        else:
            k0, k1 = tasktools.unravel_index(b-1, (len(conditions), len(in_outs)))
            cond    = conditions[k0]
            in_out = in_outs[k1]
    else:
        raise ValueError("Unknown trial type.")

    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------


    if catch_trial:
        epochs = {'T': 2500}
    else:
        if params['name'] == 'test':
            fixation = 300
        else:
            fixation = 100
        f1 = 100
        delay = 200
        f2       = 100
        decision = 200
    # else:
    #     if params['name'] == 'test':
    #         firstInterval = 300
    #     else:
    #         firstInterval = 100
    #     secondInterval = 200
    #     decision = 200
        T        = fixation + f1 + delay + f2 + decision

        epochs = {
            'fixation': (0, fixation),
            'f1':       (fixation, fixation + f1),
            'delay':    (fixation + f1, fixation + f1 + delay),
            'f2':       (fixation + f1 + delay, fixation + f1 + delay + f2),
            'decision': (fixation + f1 + delay + f2, T)
            }
        epochs['T'] = T

    #-------------------------------------------------------------------------------------
    # Trial info
    #-------------------------------------------------------------------------------------

    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, e is array of epochs w/ points
    trial = {'t': t, 'epochs': epochs}           # Trial

    if catch_trial:
        trial['info'] = {}
    else:
        # Correct choice
        
        if cond in [(yes, yes),(no, no)]:
            choice=0
        else:
            choice=1
        
        #if in_out == 'equal':
        #    choice = 0
        #else:
        #    choice = 1

        # Trial info
        trial['info'] = {'cond': cond, 'in_out': in_out, 'choice': choice}

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    X = np.zeros((len(t), Nin))
    if not catch_trial:
        # Stimulus 1
        X[e['f1'],POS] = cond[0]
        X[e['f1'],NEG] = yes + no - cond[0] #flipped frequency 1->0.2, 0.2 -> 1

        # Stimulus 2
        X[e['f2'],POS] = cond[1]
        X[e['f2'],NEG] = yes + no - cond[1] #flipped frequency 1->0.2, 0.2 -> 1
    trial['inputs'] = X

    #-------------------------------------------------------------------------------------
    # Target output
    #-------------------------------------------------------------------------------------

    if params.get('target_output', False):
        Y = np.zeros((len(t), Nout)) # Output
        M = np.zeros_like(Y)         # Mask

        # Hold values
        hi = 1
        lo = 0.2

        if catch_trial:
            Y[:] = lo
            M[:] = 1
        else:
            # Fixation
            Y[e['fixation'],:] = lo

            # Decision
            Y[e['decision'],choice]   = hi # One neuron spikes high
            Y[e['decision'],1-choice] = lo # The other spikes low

            # Only care about fixation and decision periods
            M[e['fixation']+e['decision'],:] = 1

        # Outputs and mask
        trial['outputs'] = Y
        trial['mask']    = M

    #-------------------------------------------------------------------------------------

    return trial

# Performance measure
performance = tasktools.performance_2afc_min_condition


min_error = 0.05
# Termination criterion
TARGET_PERFORMANCE = 85
def terminate(performance_history):
    return np.mean(performance_history[-1:]) >= TARGET_PERFORMANCE

# Validation
n_validation = 100*(nconditions + 1)
