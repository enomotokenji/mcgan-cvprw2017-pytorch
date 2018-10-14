import Config as Cfg
from Clouds import CloudChunk, CloudManager

import pyglet
from pyglet.gl import *
from pyglet.graphics import Batch

class CloudRenderer(CloudChunk):
	def __init__(self, X, Generator):
		super(CloudRenderer, self).__init__(X, Generator)

		self.Batch = Batch()


	def GenerateFinshed(self):
		super(CloudRenderer, self).GenerateFinshed()

		self.Batch.add(self.Length, GL_POINTS, None,
			('v2i/static', self.Points),
			('c4f/static', self.Colours)
		)

	def Draw(self, X):
		super(CloudRenderer, self).Draw(X)
		self.Batch.draw()

class CloudRenderManager(CloudManager):
	def __init__(self):
		super(CloudRenderManager, self).__init__(CloudRenderer)


class GameWindow(pyglet.window.Window):
	def __init__(self, *args, **kwargs):
		super(GameWindow, self).__init__(*args, **kwargs)

		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glPointSize(Cfg.PixelSize)

		Width, Height = self.get_size()
		glViewport(0, 0, Width, Height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0, Width, Height, 0, 1, -1)

		self.Width = float(Width)
		self.Height = Height

		self.Clouds = CloudRenderManager()

		self.XPos = 0.0
		self.XChange = 2.0

		pyglet.clock.schedule_interval(self.Update, 1.0 / Cfg.Framerate)

	def on_draw(self):
		self.clear()
		glLoadIdentity()

		glTranslatef(-self.XPos, 0, 0)
		X = int(self.XPos / Cfg.CloudWidth)
		for CloudX in xrange(X, X + int(round(self.XPos / Cfg.CloudWidth)) + 3):
			self.Clouds.GetObject(CloudX * Cfg.CloudWidth).Draw(self.XPos)



	def Update(self, dt):
		self.XPos += self.XChange

W= GameWindow()
pyglet.app.run()