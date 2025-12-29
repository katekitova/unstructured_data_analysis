# распознавание цифр
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense
import pygame
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # загрузка выборки
# далее отображение первых 25 изображений из обучающей выборки - можно убрать
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[100 + i], cmap=plt.cm.binary)

# plt.show()
# это самое главное, это сеть. здесь можно менять
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),  # потому что размер изображения 28*28
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')])
# print(model.summary())     # вывод структуры НС в консоль
x_train = x_train / 255  # нормализация
x_test = x_test / 255
y_train_cat = keras.utils.to_categorical(y_train, 10)  # приведение к вектору из 0 и 1 (10 - размерность)
y_test_cat = keras.utils.to_categorical(y_test, 10)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train_cat, batch_size=32, epochs=10, validation_split=0.2, verbose=False)  # обучение
print('*******   loss и accuracy  на тестовом наборе   ***********')
model.evaluate(x_test,
               y_test_cat)  # Метод evaluate прогоняет все тестовое множество и вычисляет значение критерия качества и метрики
# print('*********    *****************')
# print('проверка правильности некоторой цифры')
# n = 5  # изображение - это номер в наборе данных
# x = np.expand_dims(x_test[n], axis=0)
# # выведем рукописную цифру с заданным номером, полученное сетью значение - в виде числа и в виде вектора
# res = model.predict(x)
# print('печать выходного вектора для', y_test[n])
# print(res)
# print
# print(np.argmax(res), '  - такая цифра должна быть на рисунке. Верно?')
# plt.imshow(x_test[n], cmap=plt.cm.binary)
# plt.show()
# # выделение неверных результатов
# pred = model.predict(x_test)
# pred = np.argmax(pred, axis=1)  # массив предсказанных результатов (на тестовом наборе)
#
# print('вывод некоторых ошибочных некоторых результатов')
# print('смотрим, правильные ли ошибочные картинки, хорошо ли нарисованы')
#
# er = 0
# for i in range(10000):  # проверяем тестовой набор
#     if pred[i] != y_test[i]:
#         er = er + 1  # считаем ошибки
#         if er > 10 and er < 16:  # смотрим 5 картинок неверно определенных сетью
#             print("Значение сети: ", y_test[i], ' предсказано   ', pred[i])
#             plt.imshow(x_test[i], cmap=plt.cm.binary)
#             plt.show()
#
# print('k-vo  error ', er)

pygame.init()
WIDTH, HEIGHT = 600, 400
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Нарисуй цифру (SPACE - распознать, C - очистить, ESC - выйти)")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
WINDOW.fill(WHITE)

drawing = False
brush_radius = 12
font = pygame.font.SysFont(None, 32)
small_font = pygame.font.SysFont(None, 24)

result_text = ""
conf_text = ""
clock = pygame.time.Clock()
running = True

def predict_from_surface(surface):
    data_str = pygame.image.tostring(surface, "RGB")
    img = Image.frombytes("RGB", (WIDTH, HEIGHT), data_str)
    img = img.convert("L")
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32")
    arr = 255.0 - arr
    arr = arr / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    preds = model.predict(arr, verbose=False)[0]
    digit = np.argmax(preds)
    confidence = preds[digit]
    return digit, confidence

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_c:
                WINDOW.fill(WHITE)
                result_text = ""
                conf_text = ""
            elif event.key == pygame.K_SPACE:
                digit, conf = predict_from_surface(WINDOW)
                result_text = f"Модель думает: {digit}"
                conf_text = f"Уверенность: {conf*100:.1f}%"

    if drawing:
        x, y = pygame.mouse.get_pos()
        pygame.draw.circle(WINDOW, BLACK, (x, y), brush_radius)
    if result_text:
        text_surf = font.render(result_text, True, (255, 0, 0))
        WINDOW.blit(text_surf, (10, 10))
    if conf_text:
        conf_surf = small_font.render(conf_text, True, (0, 0, 255))
        WINDOW.blit(conf_surf, (10, 50))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
