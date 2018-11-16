#%%
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def measure(func):
    def wrapper(*args, **kwargs):
        start = timer()
        result = func(*args, **kwargs)
        end = timer()
        return (end - start), result
    return wrapper

@measure
def norm_two_loop(X, Y):
    dists = np.empty((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            dists[i, j] = np.sqrt(np.sum((X[i] - Y[j]) ** 2))
    return dists

@measure
def norm_one_loop(X, Y):
    dists = np.empty((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        dists[i, :] = np.sqrt(np.sum((X[i] - Y) ** 2, axis=1))
    return dists

@measure
def norm_no_loop(X, Y):
    X_sqr = np.sum(X ** 2, axis=1)
    Y_sqr = np.sum(Y ** 2, axis=1)
    dists = np.sqrt(X_sqr[:, np.newaxis] - 2.0 * X.dot(Y.T) + Y_sqr)
    return dists

#%%
def compare_perf(X, Y):
    print('Inpute shapes: X:{}, Y:{}'.format(X.shape, Y.shape))
    t2, d2 = norm_two_loop(X, Y)
    print('Two loop {}'.format(t2))

    t1, d1 = norm_one_loop(X, Y)
    print('One loop {}'.format(t1))

    t0, d0 = norm_no_loop(X, Y)
    print('No loop {}'.format(t0))

    return t2, t1, t0

    # diff = np.linalg.norm(d2 - d1, ord='fro')
    # print(diff)

col_size = 1024
X = np.random.randn(4096, col_size)
Y = np.random.randn(128, col_size)
compare_perf(X, Y)

#%%
ts0 = []
ts1 = []
ts2 = []
Xs = []
for y in np.linspace(20, 40, 10, dtype=int):
    Ys = Y[:y, :]
    Xs.append(y)
    # t2, d2 = norm_two_loop(X, Ys)
    t1, d1 = norm_one_loop(X, Ys)
    # t0, d0 = norm_no_loop(X, Ys)
    # ts2.append(t2)
    ts1.append(t1)
    # ts0.append(t0)
    # print(y, t2, t1, t0)

plt.title('ccc')
# plt.plot(Xs, ts0, label='no loop')
plt.plot(Xs, ts1, label='one loop')
# plt.plot(Xs, ts2, label='two loops')
plt.legend(loc='upper left', frameon=False)

#%%
z = np.random.randn(32, 3072)
z.nbytes


#%%
x = np.power(2, np.arange(7, 13))
# plt.xlim(0, 3000)
# plt.ylim(0, 0.01)
for i in x:
    elapsed_times = []
    z = np.linspace(1, 4096, 100, dtype=int)
    for j in z:
       X = np.random.rand(i, j)
       ts = timer()
       dists = np.sqrt(np.sum((X - X[0]) ** 2, axis=1))
       te = timer()
    #    print('{}, {} time: {}'.format(i, j, te - ts))
       elapsed_times.append(te - ts)
    plt.plot(z, elapsed_times, label='{}'.format(i))
plt.legend(loc='upper left', frameon=False)

#%%
X = 
compare_perf(X, Y)