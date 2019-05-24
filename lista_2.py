from libs.utility import *
import cv2

def run():
    ex = input("Input the exercise to execute: ")

    if ex == "2":
        ex2()
    elif ex == "5":
        ex5()


def ex2():
    img = cv2.imread("image/remedio.png", cv2.IMREAD_GRAYSCALE)
    shanon_ex(img)

def ex5():
    img = cv2.imread("image/remedio.png", cv2.IMREAD_GRAYSCALE)
    tsallis_ex(img, 0.001)
