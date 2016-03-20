from __future__ import division

import numpy as np

import imp

from pycog import tasktools

from pycog          import RNN
from pycog.figtools import Figure

m = imp.load_source('model', 'workingMemory.py')

rng = np.random.RandomState(1005)
rnn  = RNN('workingMemory.pkl', {'dt': 2})



trial_args = {'name':  'test', 'catch': False, 'pair': (30, 15)}

info = rnn.run(inputs=(m.generate_trial, trial_args), rng=rng)

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
#plot.plot(t, rnn.u[1], color=Figure.colors('blue'), lw=0.5)
plot.plot(t, rnn.z[0], color=Figure.colors('red'), lw=0.5)
plot.plot(t, rnn.z[1], color=Figure.colors('green'), lw=0.5)

plot.xlabel('time')
plot.ylabel('output')

fig.save(path='.', name='workingMemory_Figure')
