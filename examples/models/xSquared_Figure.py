from __future__ import division
import numpy as np
from pycog import tasktools

from pycog          import RNN
from pycog.figtools import Figure

tau = 100
dt  = 20

period = 8*tau

epochs = {'T': 2*period}

t, e  = tasktools.get_epochs_idx(dt, epochs)

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