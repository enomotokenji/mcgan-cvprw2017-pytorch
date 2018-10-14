Python-Clouds
=============

Python Cloud generation with Pyglet and Perlin Noise.

Explaination
------------
This works with Perlin Noise. It simply passes the X and Y coordinates through Perlin Noise. It takes this, and fades the clouds out from the edge.

Clouds are generated in chunks, each in a separate thread.

Playing Around
--------------
If you need to change settings there is a `Config.py` file:

 - __CloudHeight:__ The max height of the clouds to generate.
 - __CloudWidth:__ The width of each 'Cloud chunk'.
 - __PixelSize:__ The size of each pixel. A higher value means better rendering but lower quality clouds.
 - __NoiseOffset:__ The amount to add to noise. Higher value means more and denser clouds.
 - __Seed:__ The seed to generate the clouds from.
 - __Framerate:__ The framerate to run the demo at.
