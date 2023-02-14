# Task 1.1
import re


def clean_dataset(filename):
    with open(filename, "r") as file:
        data = file.readlines()
    cleaned_data = []
    for d in data:
        # Eliminar caracteres especiales excepto números
        d = re.sub(r"[^a-zA-Z0-9\s\t]", "", d)

        # Convertir a minúsculas
        d = d.lower()
        cleaned_data.append(d)
    # Escribir los resultados en un nuevo archivo
    with open("cleaned_" + filename, "w") as file:
        for d in cleaned_data:
            file.write(d)


filename = "entrenamiento.txt"
clean_dataset(filename)
