import cv2
import numpy as np

filename = 'l3_mapping_data/camera_image_0.jpeg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray.shape)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
print(dst.shape)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.

corners = cv2.goodFeaturesToTrack(gray, 55, 0.01, 10)

corners = np.int0(corners)

img[dst>0.01*dst.max()]=[0,0,255]
# desc = []
# for x in range(0,dst.shape[1]):
#     for y in range(0,dst.shape[0]):
#         if dst[y][x]:
#             desc.append((x,y))


# for i in corners:
#     x,y = i.ravel()
#     #cv2.circle(img, (x,y),3, 255, -1)
#     cv2.circle(img,(x,y),3,255,-1)
#print(dst>0.01*dst.max())
#y_locs = np.where(np.all(dst==True, axis=0))
#x_locs = np.where(np.all(dst>0.01*dst.max(), axis=1))

#print(desc[0:10])

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
cv2.imwrite('harris_0.png',img)