import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import scipy.interpolate as interp
import scipy.ndimage as ndimage
from scipy.interpolate import interp2d

img = Image.open('twoCircles.jpg').convert('L')
img = np.array(img)
sy, sx = img.shape
aperture = 200
angle = 180
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
ax1.imshow(img, cmap='gray')
ax1.set_title('Original')

emitter, = ax1.plot(1, 1, color='red', linewidth=5)
rays, = ax1.plot(1, 1, color='blue', linewidth=1)

x, y = np.meshgrid(np.arange(-aperture, aperture), np.arange(-aperture, aperture))
scan = np.zeros((angle, 2*aperture))
ax2.set_title('Sinogram')
ax2.set_xlabel('Aperture')

for i in range(angle):
    cos = np.cos(i * np.pi / 180)
    sin = np.sin(i * np.pi / 180)
    sample = ndimage.map_coordinates(img, [x*sin + y*cos + sx/2, -x*cos + y*sin + sy/2], order=3, cval=0)
    scan[i, :] = sample.sum(axis=1)


ax2.imshow(scan, cmap='gray', aspect='auto', extent=[-aperture, aperture, angle, 0])

scan = np.roll(scan, -aperture)
fftScan = np.fft.fft(scan, axis=1)
fftScan = np.roll(fftScan, aperture)
x,y = np.meshgrid(np.arange(-aperture, aperture), np.arange(-aperture, aperture))

angle = np.arctan2(y, x) * 180 / np.pi
offset = np.multiply(np.sqrt(x**2 + y**2), np.sign(angle - 0.00001))
wrapped_angle = np.mod(np.ceil(angle) - 0.5, 180) + 0.5
wrapped_angle = np.clip(wrapped_angle, 0, 180)


F = ndimage.map_coordinates(fftScan, [wrapped_angle, offset + aperture], order=3, cval=0)
shiftF = F

shiftF = np.roll(F, -aperture, axis=(0, 1))
invertF = np.fft.ifft2(shiftF)
invertF = np.roll(invertF, aperture, axis=(0, 1))
invertF = np.real(invertF)
invertF = np.rot90(invertF, 3)
invertF = np.fliplr(invertF)

ax3.imshow(invertF, cmap='gray', vmin=-100, vmax=100, extent=[sy, 0, 0, sx])
plt.show()

