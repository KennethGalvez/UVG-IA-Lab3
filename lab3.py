import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Cargamos los datos desde el archivo .txt
data = []
with open("entrenamiento.txt", "r") as f:
    for line in f:
        label, message = line.strip().split("\t")
        data.append([label, message])

df = pd.DataFrame(data, columns=["label", "message"])

# Dividimos el dataset en training y testing
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Contamos el número de mensajes ham y spam
n_ham_messages = train[train['label'] == 'ham'].shape[0]
n_spam_messages = train[train['label'] == 'spam'].shape[0]
total_messages = n_ham_messages + n_spam_messages

# Contamos la frecuencia de cada palabra en los mensajes ham y spam
words = {}
for index, row in train.iterrows():
    message = row['message'].split(" ")
    for word in message:
        if word in words:
            words[word][row['label']] += 1
        else:
            words[word] = {'ham': 0, 'spam': 0}
            words[word][row['label']] = 1

# Calculamos la probabilidad de cada palabra en pertenecer a ham o spam
for word in words:
    words[word]['ham'] = (words[word]['ham'] + 1) / (n_ham_messages + 2)
    words[word]['spam'] = (words[word]['spam'] + 1) / (n_spam_messages + 2)

# Predecimos la clase de los mensajes en el subset de testing
predictions = []
for index, row in test.iterrows():
    prob_ham_message = 1
    prob_spam_message = 1
    message = row['message'].split(" ")
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

'''
# CON LIBRERIAS
# Evaluamos el desempeño del modelo en el subset de testing

accuracy = accuracy_score(test['label'], predictions)
print("Accuracy en el subset de testing:", accuracy)

# Evaluamos el desempeño del modelo en el subset de training
train_predictions = []
for index, row in train.iterrows():
    prob_ham_message = 1
    prob_spam_message = 1
    message = row['message'].split(" ")
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

train_accuracy = accuracy_score(train['label'], train_predictions)
print("Accuracy en el subset de training:", train_accuracy)
'''
# SIN LIBRERIAS
# Evaluamos el desempeño del modelo en el subset de testing
correct_predictions = 0
for i in range(len(test)):
    if test.iloc[i]['label'] == predictions[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(test)
print("Accuracy en el subset de testing:", accuracy)

# Evaluamos el desempeño del modelo en el subset de training
train_predictions = []
for index, row in train.iterrows():
    prob_ham_message = 1
    prob_spam_message = 1
    message = row['message'].split(" ")
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

correct_train_predictions = 0
for i in range(len(train)):
    if train.iloc[i]['label'] == train_predictions[i]:
        correct_train_predictions += 1

train_accuracy = correct_train_predictions / len(train)
print("Accuracy en el subset de training:", train_accuracy)
