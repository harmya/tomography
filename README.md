# Computed Tomography

Tomographic reconstruction is a type of inverse problem where the challenge is to yield an estimate of a specific system from a finite number of projections.
One of the coolest parts of this process is the fourier slice theorem, the 1D Fourier Transform of the projection of an object at angle $\theta$ is equal to the slice of the 2D Fourier Transform of the object at angle $\theta$.
Using this, we can take multiple projections to construct the 2D Fourier Transform of the object, and then take its inverse to get the original object.

# Example
![CAT Scan](https://github.com/harmya/tomography/blob/main/assets/dogscan.gif)
