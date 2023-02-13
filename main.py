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
n_ham_messages = len([x for x in train if x[0] == 'ham'])
n_spam_messages = len([x for x in train if x[0] == 'spam'])
total_messages = n_ham_messages + n_spam_messages

# Contamos la frecuencia de cada palabra en los mensajes ham y spam
words = {}
for label, message in train:
    message = message.split(" ")
    for word in message:
        if word in words:
            words[word][label] += 1
        else:
            words[word] = {'ham': 0, 'spam': 0}
            words[word][label] = 1

# Calculamos la probabilidad de cada palabra en pertenecer a ham o spam
for word in words:
    words[word]['ham'] = (words[word]['ham'] + 1) / (n_ham_messages + 2)
    words[word]['spam'] = (words[word]['spam'] + 1) / (n_spam_messages + 2)

# Predecimos la clase de los mensajes en el subset de testing
predictions = []
for label, message in test:
    prob_ham_message = 1
    prob_spam_message = 1
    message = message.split(" ")
    for word in message:
        if word in words:
            prob_ham_message *= words[word]['ham']
            prob_spam_message *= words[word]['spam']
        else:
            prob_ham_message *= 1 / (n_ham_messages + 2)
            prob_spam_message *= 1 / (n_spam_messages + 2)

# Comparamos las probabilidades y clasificamos el mensaje como spam o ham
    if prob_spam_message > prob_ham_message:
        predictions.append('spam')
    else:
        predictions.append('ham')

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
            prob_ham_message *= words[word]['ham']
            prob_spam_message *= words[word]['spam']
        else:
            prob_ham_message *= 1 / (n_ham_messages + 2)
            prob_spam_message *= 1 / (n_spam_messages + 2)

    if prob_spam_message > prob_ham_message:
        train_predictions.append('spam')
    else:
        train_predictions.append('ham')

train_correct_predictions = 0
for i in range(len(train)):
    if train[i][0] == train_predictions[i]:
        train_correct_predictions += 1

train_accuracy = train_correct_predictions / len(train)
print("Accuracy en el subset de training:", train_accuracy)


# Creamos la nueva función para clasificar las palabras recibidas
def classify_message(message, words, n_ham_messages, n_spam_messages):
    prob_ham_message = 1
    prob_spam_message = 1
    message = message.split(" ")
    for word in message:
        if word in words:
            prob_ham_message *= words[word]['ham']
            prob_spam_message *= words[word]['spam']
        else:
            prob_ham_message *= 1 / (n_ham_messages + 2)
            prob_spam_message *= 1 / (n_spam_messages + 2)
    if prob_spam_message > prob_ham_message:
        return 'spam', prob_spam_message / (prob_spam_message + prob_ham_message)
    else:
        return 'ham', prob_ham_message / (prob_spam_message + prob_ham_message)


# Recibimos mensajes nuevos y los clasificamos
while True:
    message = input("\nIngrese un mensaje para definir si es spam o ham: ")
    label, prob = classify_message(
        message, words, n_ham_messages, n_spam_messages)
    print("\nEl mensaje es clasificado como: ", label)
    print("La probabilidad de ser", label, "es de: ", prob)
    continue_input = input("\n¿Desea ingresar otro mensaje? (Sí/No)").lower()
    if continue_input == "no":
        break
