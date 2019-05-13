from libs.utility import *

def run():
    ex = input("Input the exercise to execute: ")

    if ex == "2":
        ex2()
    elif ex == "7":
        print("hi")


def ex2():
    img = cv2.imread("image/japanese-temple.png", cv2.IMREAD_GRAYSCALE)
    shannon(img, 50)

