## Модели

- https://keras.io/api/applications/

| ResNet50 | VGG16 |
| :---: | :---: |
| ![ResNet50](kitchen-output-ResNet50.gif) | ![VGG16](kitchen-output-VGG16.gif) |

## Команды

Создание фреймов (изображений JPEG) из файла видео (MOV):
```
ffmpeg -i kitchen.mov -vf fps=25 kitchen/thumb%04d.jpg -hide_banner
```

Обработка одного изображения (`kitchen\thumb0004.jpg`):
```
python visualization.py --process image --path kitchen\thumb0004.jpg
```

Обработка всех фреймов видео:
```
python visualization.py --process video --path kitchen/
```

Создание видео файла (MP4) из обработанных фреймов:
```
ffmpeg -framerate 25 -i kitchen_output/result-%04d.jpg kitchen-output.mp4
```
