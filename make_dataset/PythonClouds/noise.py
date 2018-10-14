# Imports, sorted alphabetically.

# Python packages
from math import sqrt, floor
import random

# Third-party packages
from perlin import SimplexNoise

# Modules from this project
from Utils import FastRandom

# Modules from this project

__all__ = ('SimplexNoiseGen', 'PerlinNoise')

# Factory class utilizing perlin.SimplexNoise
class SimplexNoiseGen(object):
    def __init__(self, seed, octaves=6, zoom_level=0.002):  # octaves = 6,
        perm = list(range(255))
        random.Random(seed).shuffle(perm)
        self.noise = SimplexNoise(permutation_table=perm).noise2

        self.PERSISTENCE = 2.1379201 # AKA lacunarity
        self.H = 0.836281
        self.OCTAVES = octaves       # Higher linearly increases calc time; increases apparent 'randomness'
        self.weights = [self.PERSISTENCE ** (-self.H * n) for n in range(self.OCTAVES)]

        self.zoom_level = zoom_level # Smaller will create gentler, softer transitions. Larger is more mountainy

    def fBm(self,x,z):
        x *= self.zoom_level
        z *= self.zoom_level
        y = 0
        for weight in self.weights:
            y += self.noise(x, z) * weight

            x *= self.PERSISTENCE
            z *= self.PERSISTENCE
        return y


# Improved Perlin Noise based on Improved Noise reference implementation by Ken Perlin
class PerlinNoise(object):
    def __init__(self, seed):
        rand = FastRandom(seed)

        self.perm = [None] * 512
        noise_tbl = [None] * 256

        self.PERSISTENCE = 2.1379201
        self.H = 0.836281
        self.OCTAVES = 9
        self.weights = [None] * self.OCTAVES
        self.regen_weight = True

        for i in range(256):
            noise_tbl[i] = i

        for i in range(256):
            j = rand.randint() % 256
            j = abs(j)

            noise_tbl[i], noise_tbl[j] = noise_tbl[j], noise_tbl[i]

        for i in range(256):
            self.perm[i] = self.perm[i + 256] = noise_tbl[i]

    def fade(self, t):
        return (t ** 3) * (t * (t * 6 - 15) + 10)

    # linear interpolate
    def lerp(self, t, a, b):
        return a + t * (b - a)

    def grad(self, hash, x, y, z):
        h = hash & 15
        u = x if h < 8 else y
        if h < 4:
            v = y
        elif h in (12, 14):
            v = x
        else:
            v = z
        return (-u if h & 1 else u) + (-v if h & 2 else v)

    def noise(self, x, y, z):
        X = int(floor(x)) & 255
        Y = int(floor(y)) & 255
        Z = int(floor(z)) & 255

        x -= floor(x)
        y -= floor(y)
        z -= floor(z)

        u = self.fade(x)
        v = self.fade(y)
        w = self.fade(z)

        A = self.perm[X] + Y
        AA = self.perm[A] + Z
        AB = self.perm[(A + 1)] + Z
        B = self.perm[(X + 1)] + Y
        BA = self.perm[B] + Z
        BB = self.perm[(B + 1)] + Z

        return self.lerp(w,
                    self.lerp(v, self.lerp(u,
                                 self.grad(self.perm[AA], x,       y, z),
                                 self.grad(self.perm[BA], x - 1.0, y, z)),
                         self.lerp(u,
                              self.grad(self.perm[AB], x,       y - 1.0, z),
                              self.grad(self.perm[BB], x - 1.0, y - 1.0, z))),
                    self.lerp(v, self.lerp(u,
                                 self.grad(self.perm[(AA + 1)], x,       y, z - 1.0),
                                 self.grad(self.perm[(BA + 1)], x - 1.0, y, z - 1.0)),
                    self.lerp(u,
                         self.grad(self.perm[(AB + 1)], x,       y - 1.0, z - 1.0),
                         self.grad(self.perm[(BB + 1)], x - 1.0, y - 1.0, z - 1.0))))

    def fBm(self, x, y, z):
        total = 0.0

        if self.regen_weight:
            self.weights = [None] * self.OCTAVES
            for n in range(self.OCTAVES):
                self.weights[n] = self.PERSISTENCE ** (-self.H * n)

            self.regen_weight = False

        for n in range(self.OCTAVES):
            total += self.noise(x, y, z) * self.weights[n]

            x *= self.PERSISTENCE
            y *= self.PERSISTENCE
            z *= self.PERSISTENCE

        return total

    @property
    def octave(self):
        return self.OCTAVES

    @octave.setter
    def octave(self, value):
        self.OCTAVES = value
        self.regen_weight = True