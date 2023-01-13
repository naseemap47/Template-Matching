import cv2
import os


# ORB
orb = cv2.ORB_create(nfeatures=1000)

def feature_matcher(img1, img2):
    
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # BF Matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # print('Good Matches:', len(good))

    # Draw Keypoints
    imgKp1 = cv2.drawKeypoints(img1, kp1, None)
    imgKp2 = cv2.drawKeypoints(img2, kp2, None)
    out_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    return imgKp1, imgKp2, out_img, len(good)



# Master Image
# master_img = cv2.imread('images/master.jpg', 0)
# img1 = cv2.imread('images/1.jpg', 0)

# # Resize
# master_img = cv2.resize(master_img, (800, 500))
# img1 = cv2.resize(img1, (800, 500))

# Test Images
# img2 = cv2.imread('images/2.jpg')
# img3 = cv2.imread('images/3.jpg')
# img4 = cv2.imread('images/4.jpg')
# img5 = cv2.imread('images/5.jpg')

img_list = os.listdir('master')
for i in range(len(img_list)):
    
    img1 = cv2.imread(f'master/{img_list[-1]}', 0)
    img2 = cv2.imread(f'master/{img_list[i]}', 0)
    print(f'{img_list[-1]} Vs {img_list[i]}')
    
    imgKp1, imgKp2, out_img, matches = feature_matcher(img1, img2)
    print('Good Matches:', matches)


# cv2.imshow('Master Kp1', imgKp1)
# cv2.imshow('img1 Kp2', imgKp2)
# cv2.imshow('Master img', master_img)
# cv2.imshow('img1', img1)

# # out_img = cv2.resize(out_img, (800, 500))
# cv2.imshow('Output', out_img)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
