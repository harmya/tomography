import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import scipy.interpolate as interp
import scipy.ndimage as ndimage
from scipy.interpolate import interp2d
from scipy.ndimage import map_coordinates
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

img = Image.open('josh.jpeg').convert('L')
img = np.array(img)
aperture = 200
rotationRange = 181
sy, sx = img.shape
minSize = 600

if sy < minSize:
    # pad equal amount on top and bottom
    img = np.pad(img, ((minSize-sy)//2, (minSize-sy)//2), mode='constant', constant_values=0)
if sx < minSize:
    # pad equal amount on left and right
    img = np.pad(img, ((minSize-sx)//2, (minSize-sx)//2), mode='constant', constant_values=0)

sy, sx = img.shape


fig, ((ax1, ax3, ax2), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(11, 7))
ax1.imshow(img, cmap='gray')
ax1.set_title('Original')
ax2.set_title('Projection')
ax3.set_title('Sinogram')
ax4.set_title('FFT of Projection')
ax5.set_title('2D FFT from FFT of Projection')
ax6.set_title('Inverse FFT of 2D FFT')

ax1.set(xticklabels=[])
ax1.set(yticklabels=[])
ax2.set(xticklabels=[])
ax2.set(yticklabels=[])
ax3.set(yticklabels=[])
ax3.set(xticklabels=[])
ax4.set(yticklabels=[])
ax4.set(xticklabels=[])

ax6.set(yticklabels=[])
ax6.set(xticklabels=[])


emitter, = ax1.plot(1, 1, color='red', linewidth=5)
rays, = ax1.plot([1], [1], color='grey', linewidth=1)
projection, = ax2.plot([], [], color='blue', linewidth=1)
fftPlot, = ax4.plot([], [], color='black', linewidth=1)

x, y = np.meshgrid(np.arange(-aperture, aperture), np.arange(-aperture, aperture))
scan = np.zeros((rotationRange, 2*aperture))
xi = np.tile([-aperture, aperture, -aperture], 11)
yi = (np.tile(np.linspace(-1, 1, 11), 3).reshape(3, 11) * aperture)
yi = np.concatenate(yi.T)

for i in range(rotationRange):
    cos = np.cos(i * np.pi / 180)
    sin = np.sin(i * np.pi / 180)
    sample = ndimage.map_coordinates(img, [x*sin + y*cos + sx/2, -x*cos + y*sin + sy/2], order=3, cval=0)
    xd = sin * xi + cos * yi + sx/2
    yd = -cos * xi + sin * yi + sy/2
    scan[i, :] = sample.sum(axis=1)

showScan = ax3.imshow(scan, cmap='gray', aspect='auto', extent=[-aperture, aperture, rotationRange, 0])

rotatedScan = np.roll(scan, -aperture)
fftScan = np.fft.fft(rotatedScan, axis=1)
fftScan = np.roll(fftScan, aperture)
x,y = np.meshgrid(np.arange(-aperture, aperture), np.arange(-aperture, aperture))

angle = np.arctan2(y, x) * 180 / np.pi
offset = np.multiply(np.sqrt(x**2 + y**2), np.sign(angle - 0.00001))
wrapped_angle = np.mod(np.ceil(angle) - 0.5, 180) + 0.5
wrapped_angle = np.clip(wrapped_angle, 0, 180)

F = ndimage.map_coordinates(fftScan, [wrapped_angle, offset + aperture], order=3, cval=0)
shiftF = F
shiftF = np.roll(F, -aperture, axis=(0, 1))
# roll to center
plotFFt = np.roll(shiftF, aperture, axis=(0, 1))
ax5.imshow(np.log10(np.abs(plotFFt)), cmap='gray', aspect='auto')



invertF = np.fft.ifft2(shiftF)
invertF = np.roll(invertF, aperture, axis=(0, 1))
invertF = np.real(invertF)

invertF = np.rot90(invertF, 3)
invertF = np.fliplr(invertF)
ax6.imshow(invertF, cmap='gray', extent=[sy, 0, 0, sx])


def update(i):
    cos = np.cos(i * np.pi / 180)
    sin = np.sin(i * np.pi / 180)
    xd = sin * xi + cos * yi + sx/2
    yd = -cos * xi + sin * yi + sy/2

    rays.set_data(xd, yd)
    emitter.set_data([xd[0], xd[-1]], [yd[0], yd[-1]])
    projection.set_data(np.arange(2*aperture), scan[i, :])

    projAti = scan[i, :]
    fftProj = np.fft.fft(projAti)
    fftPlot.set_data(np.arange(fftProj.size), np.abs(fftProj))


    ax2.set_ylim(0, scan.max()* 1.1)
    ax2.set_xlim(0, 2*aperture)
    ax4.set_ylim(fftProj.min() * 0.9, fftProj.max() * 1.1)
    ax4.set_xlim(-5, fftProj.size + 5)
    

    temp_scan = scan.copy()
    temp_scan[i:, :] = 0
    showScan.set_array(temp_scan)

    return emitter, rays, projection, showScan, fftPlot
ani = FuncAnimation(fig, update, frames=range(rotationRange), interval=100, blit=True, save_count=200)
writervideo = animation.FFMpegWriter(fps=60) 
ani.save('josh.mp4', writer=writervideo)
plt.close()