import numpy as np
import localuf
import matplotlib.pyplot as plt

ds = range(3, 9, 2)
ps = np.linspace(0.05, 0.15, 4)
n = int(1e4)

fUF = localuf.sim.accuracy.monte_carlo(
    sample_counts={d: [(p, n) for p in ps] for d in ds},
    code_class=localuf.Surface,
    decoder_class=localuf.decoders.NN,
    noise="code capacity",
)

localuf.plot.accuracy.monte_carlo(fUF)
# localuf.plot.runtime.mean(fUF)
plt.show()
plt.grid()
