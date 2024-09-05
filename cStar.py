from Harmonium import *

mp.mp.dps = 200
nr = 0
l = 0
cut = 500
lbd = mp.mpf('1')

begin = HarmonicHelium(nr, l, lbd, 2)
gamma = begin.solve_energy(100, '3')
print(gamma)


def rule(c):
    model = HarmonicHelium(nr, l, lbd, c)
    an = model.power_series(gamma, cut)[0]
    G1 = an[0]
    G2 = 0
    for n in range(len(an)):
        G2 += an[n]
    # print(f"G1 = {G1}")
    # print(f"G2 = {G2}")
    if G1 < G2:
        return True
    else:
        return False


print(rule('1'))
print(rule('3'))

c1 = mp.mpf('1')
c2 = mp.mpf('3')

while True:
    c3 = (c1 + c2) / 2
    print(f"c = {c3}")
    if rule(c3):
        c2 = c3
    else:
        c1 = c3
    if abs(c2 - c1) < 1e-8:
        c_star = c3
        break
print(f"c_star0 = {c_star}", end='\t')
