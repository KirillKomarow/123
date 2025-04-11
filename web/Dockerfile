# Задаём базовый образ
FROM python:3.9
WORKDIR /Project_final_1/web/app
# Копируем содержимое папки ./app в рабочую директорию контейнера
COPY ./app ./
# Копируем файл requirements.txt в рабочую директорию контейнера
COPY ./requirements.txt ./
# Запускаем установку необходимых зависимостей
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python", "./server.py" ]