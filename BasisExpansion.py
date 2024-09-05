"""
File Name: BasisExpansion.py
Description: This is a Python file calculating the energy and wave function of harmonium using basis expansion method.
Author: Wenqing Yao, PKU
Last edited: 2024-09-05
Compiler Environment: python 3
"""
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


class BasisExpansion:

    def __init__(self, nr: int, l: int, lbd, alpha, cut):
        self.nr = nr
        self.l = l
        self.lbd = mp.mpf(str(lbd))
        self.alpha = mp.mpf(str(alpha))
        self.cut = cut
        self.mu = l + mp.mpf('0.5')

    def basis_function(self, n, r):
        l, mu, alpha = self.l, self.mu, self.alpha
        r = mp.mpf(str(r))
        alpha = mp.mpf(str(alpha))
        temp = mp.power(mp.pi, -1 / 2) * mp.power(2, n + l + 2) * mp.factorial(n) \
               / mp.fac2(2 * n + 2 * l + 1) * mp.power(alpha, l + 3 / 2)
        y = mp.sqrt(temp) * mp.power(r, l) * mp.laguerre(n, mu, alpha * r ** 2) * mp.exp(-alpha * r ** 2 / 2)
        return y

    def norm(self, n):
        l = self.l
        ans = mp.power(mp.pi, -1 / 2) * mp.power(2, n + l + 2) * mp.factorial(n) / mp.fac2(2 * n + 2 * l + 1)
        return mp.sqrt(ans)

    def B_matrix_element(self, n, m, gamma):
        if n > m:
            return self.B_matrix_element(m, n, gamma)
        mu = self.mu
        gamma = mp.mpf(str(gamma))
        part1 = self.norm(n) * self.norm(m) / 2
        part2 = 0
        for k in range(n + 1):
            part2 += (-1) ** (k + m) * mp.binomial(n + mu, n - k) * mp.binomial(gamma + k, m) \
                     * mp.gamma(gamma + mu + k + 1) / mp.gamma(k + 1)
        return part1 * part2

    def B_matrix_element_(self, n, m, gamma):
        # more fast when gamma is not an integer
        if n > m:
            return self.B_matrix_element(m, n, gamma)
        gamma = mp.mpf(str(gamma))
        mu = self.mu
        ans = (-1) ** m * self.norm(n) * self.norm(m) / 2 * mp.binomial(gamma, m) * mp.binomial(mu + n, n) \
              * mp.gamma(mu + gamma + 1) * mp.hyp3f2(gamma + 1, mu + gamma + 1, -n, mu + 1, gamma + 1 - m, 1)
        return ans

    def secular_matrix(self):
        alpha = self.alpha
        cut = self.cut
        lbd, l = self.lbd, self.l
        matrix = np.zeros((cut, cut), dtype=mp.mpf)
        for n in range(cut):
            for m in range(n, cut):
                temp = lbd * mp.sqrt(alpha) * self.B_matrix_element_(n, m, '-0.5')
                if m == n:
                    temp += (4 * n + 2 * l + 3) * alpha + (mp.mpf('0.25') / alpha - alpha) \
                            * self.B_matrix_element(n, m, 1)
                if m == n + 1:
                    temp += (mp.mpf('0.25') / alpha - alpha) * self.B_matrix_element(n, m, 1)
                matrix[n][m] = temp
                matrix[m][n] = temp
        return matrix

    def energy(self):
        h = self.secular_matrix()
        H = mp.matrix(h)
        E, ER = mp.eigsy(H)
        ans = E[self.nr]
        return ans

    def calc_phi(self, r_scan):
        h = self.secular_matrix()
        H = mp.matrix(h)
        E, ER = mp.eigsy(H)

        phi_scan = np.zeros_like(r_scan)
        for i in range(len(phi_scan)):
            r = r_scan[i]
            y = 0
            for m in range(self.cut):
                coef = ER[m, self.nr]
                y += coef * self.basis_function(m, r)
            phi_scan[i] = y

        return phi_scan

    def calc_residual(self, r_scan):
        l = self.l
        alpha = self.alpha
        lbd = self.lbd

        h = self.secular_matrix()
        H = mp.matrix(h)
        E, ER = mp.eigsy(H)

        epsilon = E[self.nr]
        residual_scan = np.zeros_like(r_scan)
        for i in range(len(residual_scan)):
            r = r_scan[i]
            y = 0
            for m in range(self.cut):
                coef = ER[m, self.nr]
                temp = (4 * m + 2 * l + 3) * alpha + (mp.mpf('0.25') - alpha ** 2) * r ** 2 + lbd / r - epsilon
                y += coef * temp * self.basis_function(m, r)
            residual_scan[i] = y

        return residual_scan


def test():
    nr = 0
    l = 0
    lbd = 1
    alpha = 1
    cut = 10

    model = BasisExpansion(nr, l, lbd, alpha, cut)
    epsilon = model.energy()
    print(epsilon)


def draw_phi():
    nr = 0
    l = 0
    lbd = 1
    alpha = 1
    cut = 10

    model = BasisExpansion(nr, l, lbd, alpha, cut)

    num = 201
    r_scan = mp.linspace(0, 10, num)
    phi_scan = model.calc_phi(r_scan)

    x = [float(i) for i in r_scan]
    y = [float(i) for i in phi_scan]

    plt.plot(x, y)
    plt.show()
    pass


def draw_residual():
    nr = 0
    l = 0
    lbd = 1
    alpha = 1
    cut = 30

    model = BasisExpansion(nr, l, lbd, alpha, cut)

    num = 501
    delta = 1e-4
    r_scan = mp.linspace(0 + delta, 10 + delta, num)
    residual_scan = model.calc_residual(r_scan)

    x = [float(i) for i in r_scan]
    y = [float(i) for i in residual_scan]

    plt.plot(x, y)
    plt.ylim(-0.01, 0.01)
    plt.show()
    pass


def main():
    mp.mp.dps = 30

    test()
    draw_phi()
    draw_residual()


if __name__ == '__main__':
    main()
