"""
File Name: Harmonium.py
Description: This is the main Python file calculating the energy and wave function of harmonium with modulator function G.
Author: Wenqing Yao, PKU
Last edited: 2024-09-05
Compiler Environment: python 3
"""

import mpmath as mp
import numpy as np

one = mp.mpf('1')


def square_normalize(r_scan, f_calc, equidistance=True):
    r_scan, f_calc = np.array(r_scan), np.array(f_calc)
    f2_calc = f_calc ** 2 * r_scan ** 2
    norm_coef = 0
    if equidistance:
        dr = r_scan[1] - r_scan[0]
        norm_coef = (sum(f2_calc) - (f2_calc[0] + f2_calc[-1]) / 2) * dr
    else:
        for i in range(len(r_scan) - 1):
            norm_coef += (f2_calc[i] + f2_calc[i + 1]) / 2 * (r_scan[i + 1] - r_scan[i])
    return 1 / mp.sqrt(norm_coef)


class HarmonicHelium:

    def __init__(self, nr: int, l: int, lbd, c):
        self.nr = int(nr)
        self.l = int(l)
        self.lbd = mp.mpf(str(lbd))
        self.c = mp.mpf(str(c))
        self.gamma = one

    def power_series(self, gamma, cut):
        """
        calculate the Taylor coefficients G_k's and its derivative towards gamma

        Args:
            gamma (mpf): E - l - 3/2
            cut (int): Truncation numbers in G's Taylor expansion

        Returns:
            (Gk, deriv_Gk) (tuple): A tuple containing Taylor coefficients G_k's and its derivative deriv_Gk, each is
            a NumPy array
        """
        lbd, c, l = self.lbd, self.c, self.l
        gamma = mp.mpf(str(gamma))
        p1 = [0, 1, -3, 3, -1]
        p2 = [2 * l + 2, -6 + 2 * gamma - 4 * l, 6 - 4 * gamma + 2 * l - c ** 2, -2 + 2 * gamma]
        p3 = [2 * gamma * (l + 1) - c * lbd, gamma * (c ** 2 - 3 - 2 * l + gamma), gamma * (1 - gamma)]
        deriv_p2 = [0, 2, -4, 2]
        deriv_p3 = [2 * l + 2, c ** 2 + 2 * gamma - 2 * l - 3, 1 - 2 * gamma]

        Gk = np.ones(cut, dtype=mp.mpf)
        deriv_Gk = np.zeros(cut, dtype=mp.mpf)
        Gk[1] = c * lbd / (2 * l + 2) - gamma
        deriv_Gk[1] = -one

        for k in range(1, cut - 1):
            temp1, temp2 = 0, 0
            for i in range(2, min(k + 1, len(p1))):
                temp1 += p1[i] * (k - i + 2) * (k - i + 1) * Gk[k - i + 2]
                temp2 += p1[i] * (k - i + 2) * (k - i + 1) * deriv_Gk[k - i + 2]
            for i in range(1, min(k + 1, len(p2))):
                temp1 += p2[i] * (k - i + 1) * Gk[k - i + 1]
                temp2 += deriv_p2[i] * (k - i + 1) * Gk[k - i + 1] + p2[i] * (k - i + 1) * deriv_Gk[k - i + 1]
            for i in range(0, min(k + 1, len(p3))):
                temp1 += p3[i] * Gk[k - i]
                temp2 += deriv_p3[i] * Gk[k - i] + p3[i] * deriv_Gk[k - i]
            Gk[k + 1], deriv_Gk[k + 1] = -temp1 / (k + 1) / (2 * l + 2 + k), -temp2 / (k + 1) / (2 * l + 2 + k)
        return Gk, deriv_Gk

    def newton_iteration(self, cut, gamma0, prec=20):
        """
        Calculate the energy E under a fixed truncation number cut using Newton's iteration method

        Args:
            cut (int): Truncation numbers in G's Taylor expansion
            gamma0 (mpf): Initial guess of gamma
            prec (int): convergence criterion of Newton's iteration method

        Returns:
            gamma (mpf): the final solution of truncated boundary conditions
        """
        gamma0 = mp.mpf(str(gamma0))
        epsN, epsE = mp.mpf('1E-%d' % prec), mp.mpf('1E-%d' % prec)
        lbd, c = self.lbd, self.c

        gamma_iter = [0, gamma0]

        n_iter = 0
        p = 1e10
        while abs(gamma_iter[-1] - gamma_iter[-2]) > epsE or abs(p) > epsN:
            n_iter += 1
            old_gamma = gamma_iter[-1]
            (Gk, deriv_Gk) = self.power_series(old_gamma, cut)
            p, dp = 0, 0
            for k in range(len(Gk)):
                p += (k - old_gamma + lbd / c) * Gk[k]
                dp += (k - old_gamma + lbd / c) * deriv_Gk[k] - Gk[k]
            new_gamma = gamma_iter[-1] - p / dp
            gamma_iter.append(new_gamma)
            # print("cut = {}, iter = {}, gamma = {}".format(cut, n_iter, new_gamma))
        return gamma_iter[-1]

    def solve_energy(self, prec=20, gamma0=''):
        """
        Calculate the energy gamma with a certain level of accuracy, the truncation number continues to
        increase until the final result of function "newton_iteration" converges.

        Args:
            gamma0 (mpf): Initial guess of gamma, if not given a default value will be used.
            prec (int): convergence criterion.

        Returns:
            gamma (mpf): the final solution of boundary conditions
        """
        epsN, epsE = mp.mpf('1E-%d' % prec), mp.mpf('1E-%d' % prec)
        lbd, c = self.lbd, self.c
        if gamma0 == '':
            gamma0 = self.nr + mp.sqrt(lbd)
        else:
            gamma0 = mp.mpf(gamma0)
        gamma_iter = [0, gamma0]
        cut = 0

        while abs(gamma_iter[-1] - gamma_iter[-2]) > epsE:
            cut += 10 + 20 * (cut // 100)
            n_iter = 0
            p = 1e10
            gamma = [0, gamma_iter[-1]]
            while abs(gamma[-1] - gamma[-2]) > epsE or abs(p) > epsN:
                n_iter += 1
                old_gamma = gamma[-1]
                (Gk, deriv_Gk) = self.power_series(old_gamma, cut)
                p, dp = 0, 0
                for k in range(len(Gk)):
                    p += (k - old_gamma + lbd / c) * Gk[k]
                    dp += (k - old_gamma + lbd / c) * deriv_Gk[k] - Gk[k]
                new_gamma = gamma[-1] - p / dp
                gamma.append(new_gamma)
                print("cut = {}, iter = {}, gamma = {}".format(cut, n_iter, new_gamma))
            gamma_iter.append(gamma[-1])
        self.gamma = gamma_iter[-1]

        return gamma_iter[-1]

    def calc_G(self, gamma, cut, x):
        x = mp.mpf(str(x))
        Gk = self.power_series(gamma, cut)[0]
        ans = Gk[0]
        for i in range(1, len(Gk)):
            ans += Gk[i] * mp.power(x, i)
        return ans

    def calc_phi(self, gamma, cut, r):
        r = mp.mpf(str(r))
        x = r / (r + self.c)
        G = self.calc_G(gamma, cut, x)
        ans = mp.power(r, self.l) * mp.power(1 + r / self.c, gamma) * mp.exp(-r ** 2 / 4) * G
        return ans

    def calc_deriv_phi(self, gamma, cut, r):
        r = mp.mpf(str(r))
        dr = mp.mpf(1e-20)
        y1 = self.calc_phi(gamma, cut, r - dr)
        y2 = self.calc_phi(gamma, cut, r)
        y3 = self.calc_phi(gamma, cut, r + dr)

        d1_G = (y3 - y1) / dr / 2
        d2_G = (y3 + y1 - 2 * y2) / dr ** 2
        return d1_G, d2_G

    def calc_residual(self, gamma, cut, r):
        phi = self.calc_phi(gamma, cut, r)
        d1_phi, d2_phi = self.calc_deriv_phi(gamma, cut, r)
        residual = -d2_phi - 2 / r * d1_phi + (r ** 2 / 4 + self.lbd / r - gamma - self.l - mp.mpf('1.5')) * phi
        return residual


def test():
    nr = 0
    l = 0
    lbd = mp.mpf('1')
    c = mp.mpf('2')

    model = HarmonicHelium(nr, l, lbd, c)

    model.solve_energy(50)


def main():
    mp.mp.dps = 100
    print("hello world!")
    test()


if __name__ == '__main__':
    main()
