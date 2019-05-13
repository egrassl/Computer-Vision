from libs import segmentation
from libs import fourrier_transform
from libs import filters


def run():
    opt = 1

    while opt > 0 and opt <= 10:

        print("1 - Adaptative Threshold Segmentation")
        print("2 - Simple Threshold Segmentation")
        print("3 - K-means Segmentation")
        print("4 - Fourrier Transform")
        print("5 - Average Filter")
        print("6 - Median Filter")
        print("7 - Sobel Filter")
        print("8 - Low-Pass N times")
        opt = int(input("Enter the experiment you want to se based on the labels above: "))

        if opt == 1:
            segmentation.adaptative_filter("image/book-page.jpg", (700, 950), 0, 15, 4)
            segmentation.adaptative_filter("image/kindle-page.jpg", (700, 500), 3, 7, 4)
        elif opt == 2:
            segmentation.limiar_filter("image/book-page.jpg", (700, 950), 0, 140)
            segmentation.limiar_filter("image/kindle-page.jpg", (700, 500), 3, 140)
        elif opt == 3:
            k = int(input("Select the number of clusters: "))
            segmentation.kmeans_segmentation("image/yellow-wall.jpeg", (700, 500), 3, k)
            k = int(input("Select the number of clusters: "))
            segmentation.kmeans_segmentation("image/kindle-page.jpg", (700, 500), 3, k)
            k = int(input("Select the number of clusters: "))
            segmentation.color_kmeans_segmentation("image/fruits.png", (300, 300), 5, k)
        elif opt == 4:
            fourrier_transform.FFT_Exercice("image/kindle-page.jpg")
        elif opt == 5:
            filters.mean_filter(image_path="image/kindle-page.jpg", kernel=(7, 7))
        elif opt == 6:
            filters.median_filter(image_path="image/kindle-page.jpg", kernel=7)
        elif opt == 7:
            filters.sobel_filter(image_path="image/kindle-page.jpg", kernel=7)
        elif opt == 8:
            filters.low_pass_n_times("image/kindle-page.jpg", (3,3), 5000)
