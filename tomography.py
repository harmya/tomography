import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import scipy.interpolate as interp
import scipy.ndimage as ndimage
from scipy.interpolate import interp2d
from scipy.ndimage import map_coordinates
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

aperture = 225
rotationRange = 181
minSize = 600

def load_image(path):
    img = Image.open(path).convert('L')
    img = np.array(img)
    
    if img.shape[0] < minSize:
        img = np.pad(img, ((minSize-sy)//2, (minSize-sy)//2), mode='constant', constant_values=0)
    if img.shape[0] < minSize:
        img = np.pad(img, ((minSize-sx)//2, (minSize-sx)//2), mode='constant', constant_values=0)
    
    sy, sx = img.shape

    return img, sx, sy

def init_plots():
    fig, ((ax1, ax3, ax2), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(11, 7))
    ax1.imshow(img, cmap='gray')

    ax1.set_title('Original')
    ax2.set_title('Projection')
    projectionTitle = ax2.text(0.5, 0.90, "", transform=ax2.transAxes, ha="center")
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
    ax5.set(yticklabels=[])
    ax5.set(xticklabels=[])
    ax6.set(yticklabels=[])
    ax6.set(xticklabels=[])


    emitter, = ax1.plot(1, 1, color='red', linewidth=5)
    rays, = ax1.plot([1], [1], color='grey', linewidth=1)
    projection, = ax2.plot([], [], color='blue', linewidth=1)
    fftPlot, = ax4.plot([], [], color='green', linewidth=1)
    fft2dLineRotate, = ax5.plot([], [], color='red', linewidth=2)
    fft2dMarker, = ax5.plot([], [], 'bo', markersize=5)

    return fig, ax1, ax2, ax3, ax4, ax5, ax6, emitter, rays, projection, fftPlot, fft2dLineRotate, fft2dMarker, projectionTitle


def compute_scan():
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
    
    return scan

def compute_fft(scan):
    rotatedScan = np.roll(scan, -aperture, axis=1)
    fftScan = np.fft.fft(rotatedScan, axis=1)
    fftScan = np.roll(fftScan, aperture, axis=1)

    x,y = np.meshgrid(np.arange(-aperture, aperture), np.arange(-aperture, aperture))
    angle = np.arctan2(y, x) * 180 / np.pi
    offset = np.multiply(np.sqrt(x**2 + y**2), np.sign(angle - 0.000001))
    wrapped_angle = np.mod(np.ceil(angle) - 0.5, 180) + 0.5
    wrapped_angle = np.clip(wrapped_angle, 0, 180)
    F = ndimage.map_coordinates(fftScan, [wrapped_angle, offset + aperture], order=3, cval=0)
    F_shifted = np.roll(F, -aperture, axis=(0, 1))
    inverseFFT = np.fft.ifft2(F_shifted)
    inverseFFT = np.roll(inverseFFT, aperture, axis=(0, 1))
    return F, inverseFFT

def compute_real(inverseFFT):
    real = np.real(inverseFFT)
    real = np.clip(real, 0, 255)
    real = np.rot90(real, 3)
    real = np.fliplr(real)
    return real

def compute_masked_fft(F, radAngle):
    maskX, maskY = np.meshgrid(np.arange(-aperture, aperture), np.arange(-aperture, aperture))
    mask = (-np.arctan2(maskY, maskX) < radAngle - np.pi) + (np.arctan2(maskY, -maskX) < radAngle - np.pi)
    F = F * mask
    return F

def animateProcess(scan, F, inverseFFT, real, emitter, rays, projection, fftPlot, fft2dLineRotate, fft2dMarker, projectionTitle, fig, ax2, ax3, ax4, ax5, ax6):
    xi = np.tile([-aperture, aperture, -aperture], 11)
    yi = (np.tile(np.linspace(-1, 1, 11), 3).reshape(3, 11) * aperture)
    yi = np.concatenate(yi.T)


    xL = np.linspace(-aperture, aperture, 2 * aperture)
    yL = np.zeros(xL.shape)
    
    sinogram = ax3.imshow(scan, cmap='gray', aspect='auto', extent=[-aperture, aperture, rotationRange, 0])
    logPlotFFT = np.log(1.0 + np.abs(F))

    fft2dAnim = ax5.imshow(np.log(1.0 + np.abs(F)), cmap='gray', aspect='auto', extent=[-aperture, aperture, -aperture, aperture])
    maskX, maskY = np.meshgrid(np.arange(-aperture, aperture), np.arange(-aperture, aperture))
    ax6.imshow(real, cmap='gray', aspect='auto', extent=[-aperture, aperture, -aperture, aperture])

    ax2.set_xlim(-aperture, aperture)
    ax2.set_ylim(0, scan.max() * 1.4)
    ax4.set_xlim(-aperture - 5, aperture + 5)
    ax4.set_ylim(-5, 20)


    def animate(i):
        sin = np.sin(i * np.pi / 180)
        cos = np.cos(i * np.pi / 180)
        
        xRotated = xi * sin + yi * cos + sx/2
        yRotated = -xi * cos + yi * sin + sy/2

        xLRotated = xL * cos - yL * sin 
        yLRotated = yL * cos + xL * sin

        rays.set_data(xRotated, yRotated)
        emitter.set_data([xRotated[0], xRotated[-1]], [yRotated[0], yRotated[-1]])
        projection.set_data(np.arange(-aperture, aperture), scan[i, :])
        fft2dLineRotate.set_data(xLRotated, yLRotated)

        plotScan = scan.copy()
        fftOfCurrentScan = np.fft.fft(plotScan[i, :])

        plotScan[i : , :] = 0
        sinogram.set_array(plotScan)

        fftPlot.set_data(np.arange(-aperture, aperture), np.fft.fftshift(np.log(1.0 + np.abs(fftOfCurrentScan))))

        fft2dAnim.set_array(compute_masked_fft(logPlotFFT, i * np.pi / 180))

        projectionTitle.set_text('Projection at angle: ' + str(i) + '°')
        
        return emitter, rays, sinogram, projection, fftPlot, fft2dLineRotate, fft2dAnim, fft2dMarker, projectionTitle,

    ani = FuncAnimation(fig, animate, frames=range(rotationRange), interval=60, blit=True)
    plt.show()

def main():
    fig, ax1, ax2, ax3, ax4, ax5, ax6, emitter, rays, projection, fftPlot, fft2dLineRotate, fft2dMarker, projectionTitle = init_plots()
    scan = compute_scan()
    F, inverseFFT = compute_fft(scan)
    real = compute_real(inverseFFT)
    animateProcess(scan, F, inverseFFT, real, emitter, rays, projection, fftPlot, fft2dLineRotate, fft2dMarker, projectionTitle, fig, ax2, ax3, ax4, ax5, ax6)


if __name__ == '__main__':
    img, sx, sy = load_image('assets/brain-side.jpeg')
    main()
