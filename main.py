import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Task 1.2 y 1.3


def task1_2_3():
    # Cargamos los datos desde el archivo .txt
    data = []
    with open("cleaned_entrenamiento.txt", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                label, message = parts
                data.append([label, message])

    # Dividimos el dataset en training y testing
    split = int(len(data) * 0.8)
    train = data[:split]
    test = data[split:]

    # Contamos el número de mensajes ham y spam
    n_ham_messages = len([x for x in train if x[0] == "ham"])
    n_spam_messages = len([x for x in train if x[0] == "spam"])
    total_messages = n_ham_messages + n_spam_messages

    # Contamos la frecuencia de cada palabra en los mensajes ham y spam
    words = {}
    for label, message in train:
        message = message.split(" ")
        for word in message:
            if word in words:
                words[word][label] += 1
            else:
                words[word] = {"ham": 0, "spam": 0}
                words[word][label] = 1

    # Calculamos la probabilidad de cada palabra en pertenecer a ham o spam
    for word in words:
        words[word]["ham"] = (words[word]["ham"] + 1) / (n_ham_messages + 2)
        words[word]["spam"] = (words[word]["spam"] + 1) / (n_spam_messages + 2)

    # Predecimos la clase de los mensajes en el subset de testing
    predictions = []
    for label, message in test:
        prob_ham_message = 1
        prob_spam_message = 1
        message = message.split(" ")
        for word in message:
            if word in words:
                prob_ham_message *= words[word]["ham"]
                prob_spam_message *= words[word]["spam"]
            else:
                prob_ham_message *= 1 / (n_ham_messages + 2)
                prob_spam_message *= 1 / (n_spam_messages + 2)

        # Comparamos las probabilidades y clasificamos el mensaje como spam o ham
        if prob_spam_message > prob_ham_message:
            predictions.append("spam")
        else:
            predictions.append("ham")

    # Evaluamos el desempeño del modelo en el subset de testing
    correct_predictions = 0
    for i in range(len(test)):
        if test[i][0] == predictions[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(test)
    print("Accuracy en el subset de testing:", accuracy)

    # Evaluamos el desempeño del modelo en el subset de training
    train_predictions = []
    for label, message in train:
        prob_ham_message = 1
        prob_spam_message = 1
        message = message.split(" ")
        for word in message:
            if word in words:
                prob_ham_message *= words[word]["ham"]
                prob_spam_message *= words[word]["spam"]
            else:
                prob_ham_message *= 1 / (n_ham_messages + 2)
                prob_spam_message *= 1 / (n_spam_messages + 2)

        if prob_spam_message > prob_ham_message:
            train_predictions.append("spam")
        else:
            train_predictions.append("ham")

    train_correct_predictions = 0
    for i in range(len(train)):
        if train[i][0] == train_predictions[i]:
            train_correct_predictions += 1

    train_accuracy = train_correct_predictions / len(train)
    print("Accuracy en el subset de training:", train_accuracy)

    # Contamos la frecuencia de cada palabra en los mensajes ham y spam
    words = {}
    for label, message in train:
        message = message.split(" ")
        for word in message:
            if word in words:
                words[word][label] += 1
            else:
                words[word] = {"ham": 0, "spam": 0}
                words[word][label] = 1

    # Creamos la nueva función para clasificar las palabras recibidas
    def classify_message(message, words, n_ham_messages, n_spam_messages):
        prob_ham_message = 1
        prob_spam_message = 1
        message = message.split(" ")
        for word in message:
            if word in words:
                prob_ham_message *= words[word]["ham"]
                prob_spam_message *= words[word]["spam"]
            else:
                prob_ham_message *= 1 / (n_ham_messages + 2)
                prob_spam_message *= 1 / (n_spam_messages + 2)
        if prob_spam_message > prob_ham_message:
            return "spam", prob_spam_message / (prob_spam_message + prob_ham_message)
        else:
            return "ham", prob_ham_message / (prob_spam_message + prob_ham_message)

    # Recibimos mensajes nuevos y los clasificamos
    while True:
        message = input("\nIngrese un mensaje para definir si es spam o ham: ")
        label, prob = classify_message(
            message, words, n_ham_messages, n_spam_messages)
        print("\nEl mensaje es clasificado como: ", label)
        print("La probabilidad de ser", label, "es de: ", prob)
        continue_input = input(
            "\n¿Desea ingresar otro mensaje? (Sí/No)").lower()
        if continue_input == "no":
            break


# Task 1.4
# Cargamos los datos del archivo .txt en un dataframe
def task1_4():
    data = []
    with open("cleaned_entrenamiento.txt", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                label, message = parts
                data.append([label, message])
    df = pd.DataFrame(data, columns=["label", "message"])

    # Dividimos los datos en un conjunto de entrenamiento y un conjunto de prueba
    train, test = train_test_split(df, test_size=0.2)

    # Convertimos los mensajes en una representación numérica con CountVectorizer
    vectorizer = CountVectorizer()
    train_features = vectorizer.fit_transform(train["message"])
    test_features = vectorizer.transform(test["message"])

    # Entrenamos un modelo de Bayes multinomial con MultinomialNB
    clf = MultinomialNB()
    clf.fit(train_features, train["label"])

    # Evaluamos la precisión del modelo en el conjunto de prueba
    train_predictions = clf.predict(train_features)
    train_accuracy = (train_predictions == train["label"]).mean()
    print("Accuracy en el subset de entrenamiento:", train_accuracy)

    # Evaluamos la precisión del modelo en el conjunto de prueba
    test_predictions = clf.predict(test_features)
    test_accuracy = (test_predictions == test["label"]).mean()
    print("Accuracy en el subset de prueba:", test_accuracy)


while True:
    print("--------------------LAb_3_IA-----------------------")
    print("Menu")
    print("1. Task 1.2 y 1.3---> Construcción del modelo y Clasificación de mensajes futuros")
    print("2. Task 1.4 ---> Comparación con Librerías")
    print("3. Analisis de Comparación de Librerias\n")

    try:
        respuesta = int(input("Seleccione la opción: \n"))

        if respuesta == 1:
            task1_2_3()
        elif respuesta == 2:
            task1_4()
        elif respuesta == 3:
            print("¿Cuál implementación lo hizo mejor? ¿Su implementación o la de la librería?")
            print("Nuestra implementación")
            print("Accuracy en el subset de testing: 0.6630727762803235 / Accuracy en el subset de training: 0.7496629213483146")
            print("Libreria")
            print("Accuracy en el subset de training: 0.9930337078651685 / Accuracy en el subset de testing: 0.9820305480682839")
            print("Si comparamos el resultado de ambas implementaciones se observa como la librería tuvo mas aciertos llegando a tener casi el total de precision en ambas pruebas \n")
            print("¿Por qué cree que se debe esta diferencia?")
            print("La diferencia se debe a que para el contexto de las pruebas, la libreria es mejor que nuestra implementacion")
            print("El modelo de la libreria es mejor que nuestra implementación, ya que utiliza el algoritmo de Naive Bayes multinomial, que está diseñado específicamente para problemas de clasificación de texto. Además, la libreria utiliza la técnica de CountVectorizer para convertir los mensajes de texto en una representación numérica, lo que puede mejorar la precisión de las predicciones en comparación con nuestra implementacion, que utiliza una implementación personalizada de un clasificador de Bayes. La libreria también utiliza la división de entrenamiento y prueba integrada de scikit-learn, mientras que nuestra implementacion realiza una división manual, lo que puede resultar en una selección de datos subóptima para entrenar y evaluar el modelo.")
        else:
            print("\nOpción inválida. Intente de nuevo.+\n")
    except ValueError:
        print("\nError: seleccione una opción válida (número entero).\n")

    continuar = input("\nPresione Enter para continuar o 'q' para salir.\n")
    if continuar == "q":
        break
