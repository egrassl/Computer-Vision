from libs import segmentation

opt = 1

while opt > 0 and opt <= 10:

    print("1 - Adaptative Threshold Segmentation")
    print("2 - Simple Threshold Segmentation")
    print("2 - K-means Segmentation")
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
