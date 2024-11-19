import cv2

import mediapipe as mp

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

from datetime import datetime


def CameraBroadcast():
    # Получаем сигнал камеры
    camera = cv2.VideoCapture()

    # Вводим функцию распознаваная лица
    mp_face_detection = mp.solutions.face_detection.FaceDetection()
    # Добавляем инструменты для отрисовки распознаного лица в opencv2
    mp_drawing = mp.solutions.drawing_utils

    # https://ru.stackoverflow.com/questions/1441495/Ошибка-при-попытке-вывести-видео-с-камеры-opencv?ysclid=m3n8e1wutn13432641

    # Таким образом получается запустить отдельную камеру по usb
    camera.open(0, cv2.CAP_DSHOW) # 0 - встроенная камера пк, 1 - подключенная камера к пк

    # Проверка: Если камера не запустилась (не была найдена), выводим сообщение и выходим из приложения
    if not camera.isOpened():
        print('Camera port is not working!')
        return

    # Устанавливаем разрешение окна транслирующего изображение
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

    while True:
        if camera.isOpened():
            # Получаем изображение
            camera_working, image = camera.read()

            # Конвертируем полученное с камеры изображение из BGR в RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Распознаем лицо по изображению с камеры
            face_recognitions_results = mp_face_detection.process(image_rgb)

            if camera_working:
                if face_recognitions_results.detections:
                    for detection in face_recognitions_results.detections:
                        mp_drawing.draw_detection(image, detection)


                cv2.imshow('PassportCamera', image)

                # Выходим из приложения при нажатии на клавишу 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Создание снимка при нажатии на клавишу 'f'
                if cv2.waitKey(1) & 0xFF == ord('f'):

                    # Пример кода для crop-а лица пользователя с помощью mediapipe
                    # https://stackoverflow.com/questions/71094744/how-to-crop-face-detected-via-mediapipe-in-python

                    # Получаем размеры изображения
                    image_rows, image_cols, _ = image.shape

                    # Получаем первый распознаный результат
                    detection = face_recognitions_results.detections[0]

                    # Получаем координаты разпознаного фрагмента
                    location = detection.location_data

                    # Получаем координаты куба
                    relative_bounding_box = location.relative_bounding_box
                    rect_start_point = _normalized_to_pixel_coordinates(
                        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                        image_rows)
                    rect_end_point = _normalized_to_pixel_coordinates(
                        relative_bounding_box.xmin + relative_bounding_box.width,
                        relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                        image_rows)

                    # Название сохраняемого изображения
                    image_name = f"opencv_frame_{datetime.now().strftime("%d_%m_%Y___%H_%M")}.png"

                    # Получаем координаты по отдельности
                    xleft, ytop = rect_start_point
                    xright, ybot = rect_end_point

                    # Получаем обрезанное изображение
                    crop_img = image_rgb[ytop-200: ybot+50, xleft-100: xright+100]

                    # Переводим изображение из BGR в RGB
                    my_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

                    # Сохраняем изображение
                    cv2.imwrite(image_name, my_image)

                    # Выходим из приложения
                    break

    # Выключаем камеру
    camera.release()
    # Закрываем все окна
    cv2.destroyAllWindows()

# Точка запуска приложения
if __name__ == '__main__':
    CameraBroadcast()


