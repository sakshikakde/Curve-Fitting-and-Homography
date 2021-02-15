## The file structure is a as follows:

sakshi_hw1       
-----Code -- all .py and .ipynb files       
-----Data -- video files     
-----graphs -- all plotted graphs are saved here      
--------video1     
--------video2    

## Running the code
Change the directory to sakshi_hw1/Code

### Part 1 -  Curve fitting
Run the CurveFitting.py using the command
python3 CurveFitting.py --BasePath='/home/sakshi/courses/ENPM673/sakshi_hw1/' --VideoFilePath='/home/sakshi/courses/ENPM673/sakshi_hw1/Data/Ball_travel_10fps.mp4' --SaveFolderName='graphs/video1'
The arguments are
1)BasePath - This is the base folder path
2)VideoFilePath - By default, the path is set as /home/sakshi/courses/ENPM673/sakshi_hw1/Data/Ball_travel_2_updated.mp4
3)SaveFolderName - the path to folder where all the plots will be saved. The folder must be inside the BasePath

### Part 2 -
python3 Homography.py 
