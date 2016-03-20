"""
A parametric working memory task, loosely based on the vibrotactile delayed
discrimination task.

  Neuronal population coding of parametric working memory.
  O. Barak, M. Tsodyks, & R. Romo, JNS 2010.

  http://www.jneurosci.org/content/30/28/9424.short

"""
from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin  = 1
N    = 100
Nout = 2

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

# Input labels
POS = 0
NEG = 1

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

fpairs      = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)] #note first number is >
gt_lts      = ['>', '<']    #whether it's greater or less than
nconditions = len(fpairs)*len(gt_lts)
pcatch      = 1/(nconditions + 1)

fall = np.ravel(fpairs)
fmin = np.min(fall)
fmax = np.max(fall)

def scale_p(f):     #how far it is from the min (how large it is)
    return 0.4 + 0.8*(f - fmin)/(fmax - fmin)

def scale_n(f):     #how far it is from the max (how small it is)
    return 0.4 + 0.8*(fmax - f)/(fmax - fmin)

def generate_trial(rng, dt, params):
    #---------------------------------------------------------------------------------
    # Select task condition
    #---------------------------------------------------------------------------------

    catch_trial = False
    if params['name'] in ['gradient', 'test']:
        if params.get('catch', rng.rand() < pcatch):  #pcatch probability of catch trial
            catch_trial = True
        else:
            fpair = params.get('fpair', fpairs[rng.choice(len(fpairs))]) 
            #either the initial ones it or randomly generate them
            gt_lt = params.get('gt_lt', rng.choice(gt_lts)) #random order
    elif params['name'] == 'validation':    
        b = params['minibatch_index'] % (nconditions + 1)
        if b == 0:
            catch_trial = True
        else:
            k0, k1 = tasktools.unravel_index(b-1, (len(fpairs), len(gt_lts)))
            fpair  = fpairs[k0]
            gt_lt  = gt_lts[k1]
    else:
        raise ValueError("Unknown trial type.")

    #---------------------------------------------------------------------------------
    # Epochs
    #---------------------------------------------------------------------------------

    if catch_trial:
        epochs = {'T': 2500}
    else:
        if params['name'] == 'test':
            fixation = 500
        else:
            fixation = 100
        f1 = 500
        if params['name'] == 'test':
            delay = 3000
        else:
            delay = tasktools.uniform(rng, dt, 2500, 3500)
        f2       = 500
        decision = 300
        T        = fixation + f1 + delay + f2 + decision

        epochs = {
            'fixation': (0, fixation),
            'f1':       (fixation, fixation + f1),
            'delay':    (fixation + f1, fixation + f1 + delay),
            'f2':       (fixation + f1 + delay, fixation + f1 + delay + f2),
            'decision': (fixation + f1 + delay + f2, T)
            }
        epochs['T'] = T

    #---------------------------------------------------------------------------------
    # Trial info
    #---------------------------------------------------------------------------------

    t, e  = tasktools.get_epochs_idx(dt, epochs) # Task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    if catch_trial:
        trial['info'] = {}
    else:
        # Correct choice
        if gt_lt == '>':     # If f_1 > f_2
            f1, f2 = fpair
            choice = 0
        else:                # If f_1 < f_2, reverse
            f2, f1 = fpair
            choice = 1

        # Info
        trial['info'] = {'f1': f1, 'f2': f2, 'choice': choice}

    #---------------------------------------------------------------------------------
    # Inputs
    #---------------------------------------------------------------------------------

    X = np.zeros((len(t), Nin))
    if not catch_trial:
        # Stimulus 1
        X[e['f1'],POS] = scale_p(f1)

        # Stimulus 2
        X[e['f2'],POS] = scale_p(f2)
    trial['inputs'] = X

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    if params.get('target_output', False):
        Y = np.zeros((len(t), Nout)) # Output matrix by time
        M = np.zeros_like(Y)         # Mask matrix by time

        # Hold values
        hi = 1
        lo = 0.2

        if catch_trial:
            Y[:] = lo
            M[:] = 1
        else:
            # Fixation
            Y[e['fixation'],:] = lo #while fixating it should not be spiking

            # Decision
            Y[e['decision'],choice]   = hi  #output corresponding to choice = hi
            Y[e['decision'],1-choice] = lo  #other one goes to lo

            # Mask
            M[e['fixation']+e['decision'],:] = 1

        trial['outputs'] = Y
        trial['mask']    = M

    #---------------------------------------------------------------------------------

    return trial

# Performance measure
performance = tasktools.performance_2afc_min_condition

# Termination criterion
TARGET_PERFORMANCE = 85
def terminate(performance_history):
    return np.mean(performance_history[-1:]) >= TARGET_PERFORMANCE

# Validation dataset
n_validation = 100*(nconditions + 1)

if __name__ == '__main__':
    from pycog import Model
    
    model = Model(N=N, Nin=Nin, Nout=Nout, ei=ei, Crec=Crec, Cout=Cout,
                  generate_trial=generate_trial, 
                  n_validation=n_validation, performance=performance, terminate=terminate)
    model.train('romoOne_savefile.pkl', seed=100, recover=False)

   #-------------------------------------------------------------------------------------
   # Plot
   #-------------------------------------------------------------------------------------

    from pycog          import RNN
    from pycog.figtools import Figure

    rng = np.random.RandomState(1066)
    rnn  = RNN('romoOne_savefile.pkl', {'dt': 2})

    trial_args = {'name':  'test', 'catch': False, 'fpair': (34, 26), 'gt_lt': '>'}

    info = rnn.run(inputs=(generate_trial, trial_args), rng=rng)

    fig  = Figure()
    plot = fig.add()
    
    epochs = info['epochs']
    f1_start, f1_end = epochs['f1']
    f2_start, f2_end = epochs['f2']
    t0   = f1_start
    tmin = 0
    tmax = f2_end
    
    t     = 1e-3*(rnn.t-t0)
    delay = [1e-3*(f1_end-t0), 1e-3*(f2_start-t0)]
    yall  = []

    plot.plot(t, rnn.u[0], color=Figure.colors('orange'), lw=0.5)
    plot.plot(t, rnn.z[0], color=Figure.colors('red'), lw=0.5)
    plot.plot(t, rnn.z[1], color=Figure.colors('green'), lw=0.5)
    
    plot.xlabel('time')
    plot.ylabel('output')

    fig.save(path='.', name='romoOne_Figure')
   