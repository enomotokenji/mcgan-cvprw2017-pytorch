# fast math algorithms
class FastRandom(object):
	def __init__(self, seed):
		self.seed = seed

	def randint(self):
		self.seed = (214013 * self.seed + 2531011)
		return (self.seed >> 16) & 0x7FFF
