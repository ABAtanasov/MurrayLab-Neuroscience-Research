from __future__ import division
import numpy as np
from pycog import tasktools

N    = 100
Nout = 1

ei, EXC, INH = tasktools.generate_ei(N)

tau = 100 # Time constant
dt  = 20  # Step size -- use a smaller step size if exact amplitude is important

train_bout = True
train_brec = True

var_rec = 0.05**2

period = 8*tau

epochs = {'T': 2*period}

t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
trial = {'t': t, 'epochs': epochs}           # Trial

trial['outputs'] = 0.9*np.power(t/(2*period) , 2)[:,None]   #to matrix form

def generate_trial(rng, dt, params):        #Note all trial needs is t, epochs, outputs
    return trial

min_error = 0.07

mode         = 'continuous'
n_validation = 50

if __name__ == '__main__':
    from pycog import Model
    
    model = Model(N=N, Nout=Nout, ei=ei, tau=tau, dt=dt,
                  train_brec=train_brec, train_bout=train_bout, var_rec=var_rec,
                  generate_trial=generate_trial,
                  mode=mode, n_validation=n_validation, min_error=min_error)
    model.train('savefile.pkl', seed=100, recover=False)

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    from pycog          import RNN
    from pycog.figtools import Figure
    
    rnn  = RNN('savefile.pkl', {'dt': 0.5, 'var_rec': 0.01**2})
    info = rnn.run(T=2*period)

    fig  = Figure()
    plot = fig.add()

    plot.plot(rnn.t/tau, rnn.z[0], color=Figure.colors('blue'))
    plot.xlim(rnn.t[0]/tau, rnn.t[-1]/tau)
    plot.ylim(0, 2)

    print rnn.t[0]
    print rnn.t[-1]
    plot.plot((rnn.t/tau)[:], (0.9*np.power(rnn.t/(2*period),2))[:],color=Figure.colors('orange'))

    plot.xlabel(r'$t/\tau$')
    plot.ylabel('$\sin t$')

    fig.save(path='.', name='xSquared')