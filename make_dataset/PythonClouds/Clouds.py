from math import cos, pi
from noise import SimplexNoiseGen
from threading import Thread
from queue import Queue
from time import time
import Config

class ObjectManager(object):
	def __init__(self, ObjectClass):
		self.Object = {}
		self.ObjectClass = ObjectClass

	def GetObject(self, X):
		if X in self.Object:
			return self.Object[X]

		Obj = self.ObjectClass(X, self)
		self.Object[X] = Obj

		return Obj

class CloudManager(ObjectManager):
	def __init__(self, CloudClass = None):
		self.Noise = SimplexNoiseGen(Config.Seed)

		if CloudClass == None:
			CloudClass = CloudChunk
		super(CloudManager, self).__init__(CloudClass)


class CloudChunk(object):
	def __init__(self, XPos, Generator):
		self.X = XPos
		self.Noise = Generator.Noise
		self.Generator = Generator

		self.Finished = False

		self.Generate()

                #T = Thread(target=self.Generate)
		#T.daemon = True
		#T.start()

	def Generate(self):
		#print "Starting Generation at",self.X
		#start = time()
		Points = []
		Colours = []
		Length = 0

		PCMap = {}

		#Generation stuff
		PixelSize = Config.PixelSize

		YOffset = Config.CloudHeight / 2.0

		Noise = self.Noise
		NoiseOffset = Config.NoiseOffset

		for X in range(0, Config.CloudWidth, PixelSize):
			XOff = X+self.X

			for Y in range(0, Config.CloudHeight, PixelSize):
				Points.append(XOff)
				Points.append(Y)

				Colours.append(1)
				Colours.append(1)
				Colours.append(1)

				#Get noise, round and clamp
				NoiseGen = Noise.fBm(XOff, Y) + NoiseOffset
				NoiseGen = max(0, min(1, NoiseGen))
				
				# Fade around the edges - use cos to get better fading
				Diff = abs(Y - YOffset) / YOffset
				NoiseGen *= cos(Diff * pi / 2)
				
				Colours.append(NoiseGen)

				if NoiseGen > 0:
					PCMap[(XOff, Y)] = (1, 1, 1, NoiseGen)

				Length += 1

		#Assign variables
		self.Points = Points
		self.Colours = Colours
		self.Length = Length
		self.PCMap = PCMap

		#print "Finished Generation at", self.X
		#print "\tTook",time() - start
		self.Finished = True

	def GenerateFinshed(self):
		pass
		

	def Draw(self, X):
		if self.Finished:
			self.Finished = False
			self.GenerateFinshed()
