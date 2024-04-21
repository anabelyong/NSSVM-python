import numpy as np
from copy import deepcopy
from scipy.sparse import issparse, diags
import time

class NSSVM:
    def __init__(self):
        pass

    def fit(self, X, y, pars=None):
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        out = self._NSSVM(X, y, pars)

        # for predict
        self.w = out['w'][:-1]
        self.b = out['w'][-1]

        return out

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

    def _Fnorm(self, var):
        return np.linalg.norm(var, 2) ** 2

    #Newton method for sparse SVMs with sparsity level tuning
    def _gradient_descent(self, fun, x0, lr=0.01, tol=1e-6, max_iter=100):
        x = x0
        for _ in range(max_iter):
            grad = self._approximate_gradient(fun, x)
            x_new = x - lr * grad
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        return x

    def _approximate_gradient(self, fun, x, eps=1e-8):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = np.array(x, copy=True)
            x_minus = np.array(x, copy=True)
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (fun(x_plus) - fun(x_minus)) / (2 * eps)
        return grad

    def _NSSVM(self, X, y, pars):
        t0 = time.time()
        if issparse(X) and np.count_nonzero(X) / X.size > 0.1:
            X = X.toarray()

        if self.n < 3e4:
            Qt = np.diag(y) @ X
        else:
            Qt = diags(y) @ X

        Q = Qt.T
        Fnorm = self._Fnorm

        maxit, alpha, tune, disp, tol, eta, s0, C, c = self._get_parameters(self.m, self.n)
        if pars is not None:
            if 'maxit' in pars: maxit = pars['maxit']
            if 'alpha' in pars: alpha = pars['alpha']
            if 'disp' in pars: disp = pars['disp']
            if 'tune' in pars: tune = pars['tune']
            if 'tol' in pars: tol = pars['tol']
            if 'eta' in pars: eta = pars['eta']
            if 's0' in pars: s0 = min(self.m, pars['s0'])
            if 'C' in pars: C = pars['C']
            if 'c' in pars: c = pars['c']

        T1 = np.where(y == 1)[0]
        T2 = np.where(y == -1)[0]
        nT1 = T1.size
        nT2 = T2.size

        if nT1 < s0:
            T = np.concatenate((T1, T2[:(s0 - nT1)]))
        elif nT2 < s0:
            T = np.concatenate((T1[:(s0 - nT2)], T2))
        else:
            T = np.concatenate((T1[:int(np.ceil(s0 / 2))], T2[:int(s0 - np.ceil(s0 / 2))]))

        T = np.sort(T[:s0])
        s = s0
        b = (nT1 >= nT2) - (nT1 < nT2)
        bb = b
        w = np.zeros(self.n)
        gz = -np.ones(self.m)
        ERR = np.zeros(maxit)
        ACC = np.zeros(maxit)
        ACC[0] = 1 - np.count_nonzero(np.sign(b) - y) / self.m
        ET = np.ones(s) / C

        maxACC = 0
        flag = 1
        j = 1
        r = 1.1
        count = 1
        count0 = 2
        iter0 = -1

        for iter in range(1, maxit + 1):
            if iter == 1 or flag:
                QT = Q[:, T]
                QtT = Qt[T, :]
                yT = y[T]
                ytT = yT.T

            alphaT = alpha[T]
            gzT = -gz[T]
            alyT = -ytT @ alphaT

            err = (np.abs(Fnorm(alpha) - Fnorm(alphaT)) + Fnorm(gzT) + alyT ** 2) / (self.m * self.n)
            ERR[iter - 1] = np.sqrt(err)

            if tune and iter < 30 and self.m <= 1e8:
                stop1 = iter > 5 and err < tol * s * np.log2(self.m) / 100
                stop2 = s != s0 and np.abs(ACC[iter - 1] - np.max(ACC[:iter - 1])) <= 1e-4
                stop3 = s != s0 and iter > 10 and np.max(ACC[iter - 5:iter]) < maxACC
                stop4 = count != count0 + 1 and ACC[iter - 1] >= ACC[0]
                stop = stop1 and (stop2 or stop3) and stop4
            else:
                stop1 = err < tol * np.sqrt(s) * np.log10(self.m)
                stop2 = iter > 4 and np.std(ACC[iter - 2:iter]) < 1e-4
                stop3 = iter > 20 and np.abs(np.max(ACC[iter - 9:iter]) - maxACC) <= 1e-4
                stop = (stop1 and stop2) or stop3

            if disp:
                print(f'  {iter:3d}          {err:.2e}         {ACC[iter - 1]:.5f}')

            if ACC[iter - 1] > 0 and (ACC[iter - 1] >= 0.99999 or stop):
                break

            ET0 = deepcopy(ET)
            ET = (alphaT >= 0) / C + (alphaT < 0) / c

            if min(self.n, s) > 1e3:
                d = self._my_cg(QT, yT, ET, np.concatenate((gzT, [alyT])), 1e-10, 50, np.zeros(s + 1))
                dT = d[:s]
                dend = d[-1]
            else:
                if s <= self.n:
                    if iter == 1 or flag:
                        PTT0 = QtT @ QT
                    PTT = PTT0 + np.diag(ET)
                    d = np.linalg.solve(np.concatenate((np.concatenate((PTT, ytT.reshape(-1, 1)), axis=1),
                                                        np.concatenate((ytT.reshape(-1, 1), np.array([[0]])),
                                                                        axis=1)), axis=0), np.concatenate(
                        (gzT, [alyT])), use_umfpack=True)
                    dT = d[:s]
                    dend = d[-1]
                else:
                    ETinv = 1 / ET
                    flag1 = np.count_nonzero(ET0) != np.count_nonzero(ET)
                    flag2 = np.count_nonzero(ET0) == np.count_nonzero(ET) and np.count_nonzero(ET0 - ET) == 0
                    if iter == 1 or flag or flag1 or not flag2:
                        EQtT = diags(ETinv) @ QtT
                        P0 = np.eye(self.n) + QT @ EQtT
                    Ey = ETinv * yT
                    Hy = Ey - EQtT @ (np.linalg.solve(P0, QT @ Ey))
                    dend = (gzT @ Hy - alyT) / (ytT @ Hy)
                    tem = ETinv * (gzT - dend * yT)
                    dT = tem - EQtT @ (np.linalg.solve(P0, QT @ tem))

            alpha = np.zeros(self.m)
            alphaT = alphaT + dT
            alpha[T] = alphaT
            b = b + dend

            w = QT @ alphaT
            Qtw = Qt @ w
            tmp = y * Qtw

            gz = Qtw - 1 + b * y
            ET1 = (alphaT >= 0) / C + (alphaT < 0) / c
            gz[T] = alphaT * ET1 + gz[T]

            j = iter + 1
            ACC[j - 1] = 1 - np.count_nonzero(np.sign(tmp + b) - y) / self.m

            if self.m <= 1e7:
                bb = np.mean(yT - tmp[T])
                ACCb = 1 - np.count_nonzero(np.sign(tmp + bb) - y) / self.m
                if ACC[j - 1] >= ACCb:
                    bb = b
                else:
                    ACC[j - 1] = ACCb
            else:
                bb = b

            if self.m < 6e6 and ACC[j - 1] < 0.5:
                b0 = self._gradient_descent(lambda t: np.sum((np.sign(tmp + t) - y) ** 2), [bb])
                acc0 = 1 - np.count_nonzero(np.sign(tmp + b0) - y) / self.m
                if ACC[j - 1] < acc0:
                    bb = b0
                    ACC[j - 1] = acc0

            if ACC[j - 1] >= maxACC:
                maxACC = ACC[j - 1]
                alpha0 = alpha
                tmp0 = tmp
                maxwb = np.concatenate((w, [bb]))

            T0 = T
            mark = 0
            if tune and (err < tol or iter % 10 == 0) and iter > iter0 + 2 and count < 10:
                count0 = count
                count += 1
                s = min(self.m, np.ceil(r * s))
                iter0 = iter
                if count > (int(self.m >= 1e6) or (self.n < 3)) + 1 * (self.m < 1e6 and self.n >= 5):
                    alpha = np.zeros(self.m)
                    gz = -np.ones(self.m)
                    mark = 1
            else:
                count0 = count

            if s != self.m:
                if self.m < 5e8:
                    T = np.argsort(np.abs(alpha - eta * gz))[-s:]
                else:
                    T = np.argsort(np.abs(alpha - eta * gz))[::-1][:s]
                if mark:
                    nT = np.count_nonzero(y[T] == 1)
                    if nT == s:
                        if nT2 <= 0.75 * s:
                            T = np.concatenate((T[:s - int(nT2 / 2)], T2[:int(nT2 / 2)]))
                        else:
                            T = np.concatenate((T[:int(s / 4)], T2[:s - int(s / 4)]))
                    elif nT == 0:
                        if nT1 <= 0.75 * s:
                            T = np.concatenate((T[:s - int(nT1 / 2)], T1[:int(nT1 / 2)]))
                        else:
                            T = np.concatenate((T[:int(s / 4)], T1[:s - int(s / 4)]))
                    T = np.sort(T)

            else:
                T = np.arange(self.m)
            flag = 1
            flag3 = np.count_nonzero(T0) == s

            if flag3:
                flag3 = np.count_nonzero(T - T0) == 0
            if flag3 or np.count_nonzero(T0) == self.m:
                flag = 0
                T = T0

        wb = np.concatenate((w, [bb]))
        acc = ACC[j - 1]

        if self.m <= 1e7 and iter > 1:
            b0 = self._gradient_descent(lambda t: np.linalg.norm(np.sign(tmp0 + t[0]) - y), [maxwb[-1]])
            acc0 = 1 - np.count_nonzero(np.sign(tmp0 + b0) - y) / self.m
            if acc < acc0:
                wb = np.concatenate((maxwb[:-1], b0))
                acc = acc0

        if acc < maxACC - 1e-4:
            alpha = alpha0
            wb = maxwb
            acc = maxACC

        out = {'s': s,
               'w' : wb,
               'sv' : s,
               'ACC' : acc,
               'iter' : iter,
               'time' : time.time() - t0,
               'alpha' : alpha
            }
    

        return out
    
    def _get_parameters(self, m, n):
        maxit = 1000
        alpha = np.zeros(m)
        tune = 0
        disp = 1
        tol = 1e-6
        eta = min(1 / m, 1e-4)
        beta = 1 if max(m, n) < 1e4 else 0.05 if m <= 5e5 else 10
        s0 = int(np.ceil(beta * n * (np.log2(m / n)) ** 2))
        C = np.log10(m) if m > 5e6 else 1 / 2
        c = C / 100
        return maxit, alpha, tune, disp, tol, eta, s0, C, c

    
    def _my_cg(self, Q, y, E, b, cgtol, cgit, x):
        r = b
        e = np.sum(r * r)
        t = e
        for i in range(cgit):
            if e < cgtol * t:
                break
            if i == 1:
                p = r
            else:
                p = r + (e / e0) * p
            p1 = p[:-1]
            w = ((Q @ p1).T @ Q).T + E * p1 + p[-1] * y
            a = e / np.sum(p * w)
            x = x + a * p
            r = r - a * w
            e0 = e
            e = np.sum(r * r)
        return x
    
