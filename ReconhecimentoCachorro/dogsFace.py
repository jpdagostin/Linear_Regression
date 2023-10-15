import cv2
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

cascade_path = "C:\\Users\\dagos\\OneDrive\\Desktop\\Aprendizado_de-maquina\\Utils\\BuscaBinaria\\ReconhecimentoCachorro\\haarcascade_frontalDOGface_extended.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Inicializar o objeto HOG
hog = cv2.HOGDescriptor()

def preprocess_image(image_path):
    # Carregar a imagem em escala de cinza
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Redimensionar a imagem para um tamanho menor
    resized_image = cv2.resize(gray, (500, 500))  # Altere o tamanho desejado
    
    # Detectar faces na imagem
    faces = face_cascade.detectMultiScale(resized_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Verificar se foram encontradas faces
    if len(faces) == 0:
        return None
    
    # Extrair região das faces encontradas
    x, y, w, h = faces[0]
    face_image = gray[y:y+h, x:x+w]
    
    # Redimensionar a imagem para um tamanho fixo (opcional)
    face_image = cv2.resize(face_image, (100, 100))
    
    return face_image

# Pasta contendo as imagens do Toby
toby_folder = "C:\\Users\\dagos\\OneDrive\\Desktop\\Aprendizado_de-maquina\\Utils\\BuscaBinaria\\ReconhecimentoCachorro\\toby"

# Percorrer todas as imagens do Toby e pré-processá-las
toby_images = []
for filename in os.listdir(toby_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(toby_folder, filename)
        face_image = preprocess_image(image_path)
        if face_image is not None:
            toby_images.append(face_image)

print("Número de imagens do Toby:", len(toby_images))

# Pasta contendo as imagens não-Toby
nontoby_folder = "C:\\Users\\dagos\\OneDrive\\Desktop\\Aprendizado_de-maquina\\Utils\\BuscaBinaria\\ReconhecimentoCachorro\\cahcorro"

# Percorrer todas as imagens não-Toby e pré-processá-las
nontoby_images = []
for filename in os.listdir(nontoby_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(nontoby_folder, filename)
        face_image = preprocess_image(image_path)
        if face_image is not None:
            nontoby_images.append(face_image)

print("Número de imagens não do Toby:", len(nontoby_images))

# # Extração de características usando Histograma de Gradientes Orientados (HOG)
# def extract_features(images):
#     features = []
#     for image in images:
#         hog_features = hog.compute(image)
#         features.append(hog_features.flatten())
#     return np.array(features)

# Extrair características usando Histograma de Gradientes Orientados (HOG)
def extract_features(images):
    features = []
    for image in images:
        hog_features = hog.compute(image)
        features.append(hog_features.flatten())
    return np.array(features)

# Configurar o HOG
winSize = (64, 64)  # Ajuste o tamanho da janela
blockSize = (16, 16)  # Ajuste o tamanho do bloco
blockStride = (8, 8)  # Ajuste o stride do bloco
cellSize = (8, 8)  # Ajuste o tamanho da célula
nbins = 9  # Número de bins do histograma

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)


# Concatenar as imagens do Toby e Não Toby
all_images = toby_images + nontoby_images

# Definir rótulos para as imagens
toby_labels = np.ones(len(toby_images))
nontoby_labels = np.zeros(len(nontoby_images))
all_labels = np.concatenate((toby_labels, nontoby_labels))

# Extrair características faciais usando HOG
hog_features = extract_features(all_images)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(hog_features, all_labels, test_size=0.2, random_state=42)

# Treinamento do classificador SVM
clf = svm.SVC()
clf.fit(X_train, y_train)

# Predição dos rótulos usando o conjunto de teste
y_pred = clf.predict(X_test)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo: {:.2f}%".format(accuracy * 100))

def classify_image(image_path):
    # Pré-processamento da imagem
    face_image = preprocess_image(image_path)
    
    if face_image is None:
        print("Não foi encontrado nenhum cachorro na imagem.")
        return
    
    # Extração de características usando HOG
    hog_features = extract_features([face_image])
    
    # Classificação usando o modelo SVM treinado
    prediction = clf.predict(hog_features)
    
    if prediction[0] == 1:
        print("O cachorro na imagem é o Toby.")
    else:
        print("O cachorro na imagem não é o Toby.")

# Classificar uma nova imagem

# new_image_path = "C:\\Users\\dagos\\OneDrive\\Desktop\\Aprendizado_de-maquina\\Utils\\BuscaBinaria\\ReconhecimentoCachorro\\downloads\\cachorro\\toby6.jpg"

new_image_path = "C:\\Users\\dagos\\OneDrive\\Desktop\\Aprendizado_de-maquina\\Utils\\BuscaBinaria\\ReconhecimentoCachorro\\toby\\toby 1 (2).jpg"
#C:\\Users\\dagos\\OneDrive\\Desktop\\ReconhecimentoCachorro\\pneu.jpg
classify_image(new_image_path)
