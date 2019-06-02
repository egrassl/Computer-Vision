from libs.utility import *
import cv2
from matplotlib import pyplot as plt
from libs.regiongrowing import *
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
from sklearn import metrics
from libs.RetrievalSystem import *
from libs.RetrievalPR import *
from PIL import Image
from sklearn.naive_bayes import GaussianNB

generalist_images = {"Satelite Image": "image/Generalist961/image1.jpg", "Airplane": "image/Generalist961/image120.jpg",
                     "Elephant": "image/Generalist961/image210.jpg", "Arrow": "image/Generalist961/image221.jpg",
                     "Lion": "image/Generalist961/image300.jpg", "Gray Bear": "image/Generalist961/image400.jpg",
                     "Polar Bear": "image/Generalist961/image450.jpg", "Human": "image/Generalist961/image500.jpg",
                     "Sunset": "image/Generalist961/image600.jpg", "Wall texture": "image/Generalist961/image700.jpg",
                     "Car": "image/Generalist961/image900.jpg"}

gn = GaussianNB()
gn2 = GaussianNB()
gn3 = GaussianNB()

def run():
    ex = input("Input the exercise to execute: ")

    if ex == "2":
        ex2()
    elif ex == "3":
        ex3()
    elif ex == "5":
        ex5()
    elif ex == "6":
        ex6()
    elif ex == "7":
        ex7()
    elif ex == "9":
        ex09()
    elif ex == "10":
        ex10()
    elif ex == "11":
        ex11()
    elif ex == "12":
        ex12()
    elif ex == "13":
        ex13()
    elif ex == "14":
        ex14()
    elif ex == "15":
        ex15()
    elif ex == "16":
        ex16()


def ex2():
    img = cv2.imread("image/japanese-temple.png", cv2.IMREAD_GRAYSCALE)
    shanon_ex(img)


def ex5():
    img = cv2.imread("image/japanese-temple.png", cv2.IMREAD_GRAYSCALE)
    tsallis_ex(img, 0.8)


def ex6():
    img = cv2.imread("image/japanese-temple.png", cv2.IMREAD_GRAYSCALE)
    tsallis_ex_2x(img, 2.0)


def ex7():
    img = cv2.imread("image/japanese-temple.png", cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.medianBlur(img, 5)

    kernel = np.ones((7,7),np.uint8)

    plt.imshow(img_blur, cmap="binary")
    plt.show()

    x, thresh = tsallis(img_blur, 0.999)

    img_seg = img_blur.copy()

    for i in range(0, len(img_blur)):
        for j in range(0, len(img_blur[0])):
            img_seg[i][j] = 255 if img_seg[i][j] >= thresh else 0


    plt.imshow(img_seg, cmap="binary")
    plt.show()

    opened = cv2.morphologyEx(img_seg, cv2.MORPH_OPEN, kernel)

    plt.imshow(opened, cmap="binary")
    plt.show()


def ex09():
    img = cv2.imread("image/japanese-temple.png", cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.medianBlur(img, 5)

    kernel = np.ones((7, 7), np.uint8)

    plt.imshow(img_blur, cmap="binary")
    plt.show()

    x, thresh = tsallis(img_blur, 0.999)

    img_seg = img_blur.copy()

    for i in range(0, len(img_blur)):
        for j in range(0, len(img_blur[0])):
            img_seg[i][j] = 255 if img_seg[i][j] >= thresh else 0



    opened = cv2.morphologyEx(img_seg, cv2.MORPH_OPEN, kernel)


    contours, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(opened, contours, -1, 120, 3)

    plt.imshow(opened, cmap="plasma")
    plt.show()


def ex10():
    img = cv2.imread("image/japanese-temple7.png", cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.medianBlur(img, 5)

    kernel = np.ones((5, 5), np.uint8)

    plt.imshow(img_blur, cmap="binary")
    plt.show()

    x, thresh = tsallis(img_blur, 0.999)

    img_seg = img_blur.copy()

    for i in range(0, len(img_blur)):
        for j in range(0, len(img_blur[0])):
            img_seg[i][j] = 255 if img_seg[i][j] >= thresh else 0

    print("K = " + str(thresh))

    plt.imshow(img_seg, cmap="binary")
    plt.show()

    opened = cv2.morphologyEx(img_seg, cv2.MORPH_OPEN, kernel)

    plt.imshow(opened, cmap="binary")
    plt.show()


def ex11():
    img1 = cv2.imread("image/japanese-temple.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("image/japanese-temple7.png", cv2.IMREAD_GRAYSCALE)

    # ==== Templo 1 ====

    img1_blur = cv2.medianBlur(img1, 5)

    kernel = np.ones((7, 7), np.uint8)

    x, thresh = tsallis(img1_blur, 0.999)

    img1_seg = img1_blur.copy()

    for i in range(0, len(img1_blur)):
        for j in range(0, len(img1_blur[0])):
            img1_seg[i][j] = 255 if img1_seg[i][j] >= thresh else 0

    opened1 = cv2.morphologyEx(img1_seg, cv2.MORPH_OPEN, kernel)

    contours1, _ = cv2.findContours(opened1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # ==== Templo 2 ====

    img2_blur = cv2.medianBlur(img2, 5)

    kernel = np.ones((7, 7), np.uint8)

    x, thresh = tsallis(img2_blur, 0.999)

    img2_seg = img2_blur.copy()

    for i in range(0, len(img2_blur)):
        for j in range(0, len(img2_blur[0])):
            img2_seg[i][j] = 255 if img2_seg[i][j] >= thresh else 0

    opened2 = cv2.morphologyEx(img2_seg, cv2.MORPH_OPEN, kernel)

    contours2, _ = cv2.findContours(opened2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # ==== Medidas ====

    maxlength = max(map(len, contours1))

    cnts1= [contour[0:2] for contour in contours1]
    cnts2 = [contour[0:2] for contour in contours2]
    cnts1 = np.array(cnts1).reshape((66,2))
    cnts2 = np.array(cnts2).reshape((66,2))


    dist = directed_hausdorff(cnts1, cnts2)[0]

    print("A distância de Hausdorf é: " + str(dist))

    dist = pdm_dist(cnts1, cnts2)
    print("A distância PDM é: " + str(dist))

    dist = euclidiana(cnts1, cnts2)
    print("A distância Euclidiana é: " + str(dist))


def euclidiana(c1, c2):

    sum = np.sum([distance.euclidean(c1[i], c2[i]) for i in range(0, len(c1))])
    return sum**0.5



def ex12():
    dist = d_kl("image/japanese-temple.png", "image/japanese-temple7.png")
    print("A distância KL das duas imagens é: " + str(dist))


def ex13():
    ImageRet = RetrievalSystem("image/Generalist961", get_generalist_labels(), d_kl_ret)
    #ImageRet = RetrievalSystem("image/Generalist961", get_generalist_labels(), general_dist)
    precisions, recalls = ImageRet.retrieve_for_iamges(generalist_images.values())
    pr = Retrieval_Pr(precisions, recalls, list(generalist_images.keys()))
    #pr.plot_curves()
    pr.plot_mean_curve()


def ex14():

    ImageRet = RetrievalSystem("image/Generalist961", get_generalist_labels(), d_kl_ret_angle)
    #ImageRet = RetrievalSystem("image/Generalist961", get_generalist_labels(), general_dist)
    precisions, recalls = ImageRet.retrieve_for_iamges(generalist_images.values())
    pr = Retrieval_Pr(precisions, recalls, list(generalist_images.keys()))
    #pr.plot_curves()
    pr.plot_mean_curve()


def ex15():
    ImageRet = RetrievalSystem("image/Generalist961", get_generalist_labels(), gaussian_hist_comparison)

    images_hist = [image.histogram for image in ImageRet.images]
    images_grad = [image.gradient_histogram for image in ImageRet.images]
    images_complete = [image.histogram.tolist() + image.gradient_histogram.tolist() for image in ImageRet.images]
    image_labels = [image.label for image in ImageRet.images]

    # ====== 1 ======

    gn.fit(images_hist, image_labels)

    precisions, recalls = ImageRet.retrieve_for_iamges(generalist_images.values())

    pr = Retrieval_Pr(precisions, recalls, list(generalist_images.keys()))
    pr.plot_curves()
    pr.plot_mean_curve()


    # ====== 2 ======

    ImageRet.comp_func = gaussian_grad_comparison

    gn2.fit(images_grad, image_labels)

    precisions, recalls = ImageRet.retrieve_for_iamges(generalist_images.values())

    pr2 = Retrieval_Pr(precisions, recalls, list(generalist_images.keys()))
    pr2.plot_curves()
    pr2.plot_mean_curve()


    # ====== 3 ======

    ImageRet.comp_func = gaussian_complete_comparison

    gn3.fit(images_complete, image_labels)

    precisions, recalls = ImageRet.retrieve_for_iamges(generalist_images.values())

    pr3 = Retrieval_Pr(precisions, recalls, list(generalist_images.keys()))
    pr3.plot_curves()
    pr3.plot_mean_curve()


def gaussian_hist_comparison(image1, image2):
    va = gn.predict_proba([image2.histogram])[0].tolist()
    c_ind = np.where(gn.classes_ == image2.label)[0][0]
    return 1.0 - va[c_ind]


def gaussian_grad_comparison(image1, image2):
    va = gn2.predict_proba([image2.gradient_histogram])[0].tolist()
    c_ind = np.where(gn2.classes_ == image2.label)[0][0]
    return 1.0 - va[c_ind]


def gaussian_complete_comparison(image1, image2):
    data = image2.histogram.tolist() + image2.gradient_histogram.tolist()
    va = gn3.predict_proba([data])[0].tolist()
    c_ind = np.where(gn3.classes_ == image2.label)[0][0]
    return 1.0 - va[c_ind]


def ex16():

    tota_size = 90.0
    sizes = [1.415, 25.4, 32.2, 56.4, 60.1, 67.9, 68.0, 69.8]

    sizes = [(1 - s/tota_size) * 100 for s in sizes]

    noise = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 100.0]

    plt.title("Compression Rate")
    plt.xlabel("Noise")
    plt.ylabel("Compression Rate")
    plt.plot(noise, sizes, c='g')
    plt.show()


    #create_ex16_images()
    #create_ex16_images2()



def create_ex16_images():
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    for size in sizes:
        image = np.empty(300 * 300)
        for i in range(0, len(image)):
            image[i] = random.randint(0, size-1)

        image = image.reshape((300, 300))
        misc.imsave('image/ex16/outfile' + str(size) + '.jpg', image)


def create_ex16_images2():
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    for size in sizes:
        image = np.empty(300 * 300)
        for i in range(0, len(image)):
            image[i] = random.randint(0, size-1)

        image = image.reshape((300, 300))
        misc.imsave('image/ex16/outfile' + str(size) + '.png', image)