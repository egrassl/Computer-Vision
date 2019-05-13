import lista_1
import lista_2


lista = input("Input the list number: ")

if lista == "1":
    lista_1.run()
elif lista == "2":
    lista_2.run()
else:
    print("Invalid list number!\nTry to execute the program again")

