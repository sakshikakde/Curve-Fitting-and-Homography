# File structure
    .
    ├── Code
    |  ├── CurveFitting.py
    |  ├── Homography.py
    ├── Data
    |  ├── Video1.mp4
    |  ├── Video2.mp4
    ├── Results
    |  ├── Video1
    |  |  ├── .png files
    |  ├── Video2
    |  |  ├── .png files
    
# Problem 1
A ball is thrown against a white background and a camera sensor is used to track its
trajectory. We have a near perfect sensor tracking the ball in video1 and the second sensor is
faulty and tracks the ball as shown in video2. Clearly, there is no noise added to the first video
whereas there is significant noise in video 2. Assuming that the trajectory of the ball follows
the equation of a parabola
- Use Standard Least Squares, TLS and RANSAC methods to fit curves to the given videos
in each case. You have to plot the data and your best fit curve for each case. Submit
your code along with the instructions to run it. (Hint: Read the video frame by frame
using OpenCV’s inbuilt function. For each frame, filter the red channel for the ball and
detect the topmost and bottom most colored pixel and store it as X and Y coordinates.
Use this information to plot curves.) 
- Briefly explain all the steps of your solution and discuss which would be a better choice
of outlier rejection technique for each case.


## Running the code
- Change the directory to the root folder
- Run the following command:
``` python3 CurveFitting.py --BasePath='./' --VideoFilePath='./Data/Ball_travel_10fps.mp4' --SaveFolderName='graphs/video1' ```
## Parameters
- BasePath - This is the base folder path
- VideoFilePath - By default, the path is set as ./Data/Ball_travel_2_updated.mp4
- SaveFolderName - the path to folder where all the plots will be saved. The folder must be inside the BasePath
## Results
### Video1
#### Data points
![alt test](https://github.com/sakshikakde/Curve-Fitting-and-Homography/blob/main/graphs/video1/points.png)
#### Least Square Method
![alt test](https://github.com/sakshikakde/Curve-Fitting-and-Homography/blob/main/graphs/video1/LScurve.png)
#### Total Least Square Method
![alt test](https://github.com/sakshikakde/Curve-Fitting-and-Homography/blob/main/graphs/video1/TLScurve.png)
#### RANSAC
![alt test](https://github.com/sakshikakde/Curve-Fitting-and-Homography/blob/main/graphs/video1/RANSACcurve.png)

### Video2
#### Data points
![alt test](https://github.com/sakshikakde/Curve-Fitting-and-Homography/blob/main/graphs/video2/points.png)
#### Least Square Method
![alt test](https://github.com/sakshikakde/Curve-Fitting-and-Homography/blob/main/graphs/video2/LScurve.png)
#### Total Least Square Method
![alt test](https://github.com/sakshikakde/Curve-Fitting-and-Homography/blob/main/graphs/video2/TLScurve.png)
#### RANSAC
![alt test](https://github.com/sakshikakde/Curve-Fitting-and-Homography/blob/main/graphs/video2/RANSACcurve.png)


# Problem 2
Mathematically compute Homograpjhy matrix for given points.
## Running the code
- Change the directory to the root folder
- Run the following command:
``` python3 Homography.py 
