import cv2
import numpy as np
import matplotlib.pyplot as plt

def getBinaryImage(img):
    print("extracting info...")
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ( _ , bin_image) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    return bin_image

def extractFeatures(img):
    print(img.shape)
    
    indices = np.where(img == 0)
    x = indices[0]
    y = indices[1]
    min_arg = np.argmin(x)
    max_arg = np.argmax(x)
    co_ordinate_min = np.array([x[min_arg], y[min_arg]])  
    co_ordinate_max = np.array([x[max_arg], y[max_arg]])
    co_ordinate = (co_ordinate_min + co_ordinate_max) / 2
    return co_ordinate

def plotGraph(array):
    print("Plotting graph")
    x = array[:,0]
    y = array[:,1]
    plt.figure()
    plt.plot(x,y)
    plt.show()
    #plt.savefig("/home/sakshi/courses/ENPM673/graphs/test.png")

def convertImg2Cartesian(points_image, image_size):
    print("converting...")
    x_i = image_size[0]
    y_i = image_size[1]

    x_c = points_image[:, 1]
    y_c = points_image[:, 0]

    x_c = x_i - x_c
    points_cartesian = [x_c, y_c]
    return points_cartesian

    




def main():
    video_file = "/home/sakshi/courses/ENPM673/Data/Ball_travel_10fps.mp4"
    cap = cv2.VideoCapture(video_file)
    # img = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # extractFeatures(img)
    co_ordinate_array = []
    image_size = []

    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Stream ended...")
            break

        bin_image = getBinaryImage(frame)
        co_ordinate = extractFeatures(bin_image)
        image_size = bin_image.shape
        co_ordinate_array.append(co_ordinate)
        

        cv2.imshow('frame',bin_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    co_ordinate_array = np.array(co_ordinate_array)
    co_ordinate_array = convertImg2Cartesian(co_ordinate_array, image_size)

    # print("Plotting graph")
    # x1 = co_ordinate_array1[:,0]
    # y1 = co_ordinate_array1[:,1]
    # x2 = co_ordinate_array2[:,0]
    # y2 = co_ordinate_array2[:,1]
    # plt.figure()
    # plt.plot(y1, x1, 'r*', y2, x2, 'bo' )
    # #plt.show()
    # plt.savefig("/home/sakshi/courses/ENPM673/graphs/test.png")

    # print("array is ", co_ordinate_array1)
    # print("array is ", co_ordinate_array2)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()