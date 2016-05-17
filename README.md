# ms-2016-gauss
Фильтр Гаусса

### Задание

Разработать программу, применяющую фильтр Гаусса к указанному списку изображений.

### Требования

* Программа принимает на вход список файлов изображений и радиус размытия
* Программа должна понимать как минимум 2 формата файлов изображений
* Должна быть предусмотрена возможность добавления новых форматов
* У программы должен быть параметр для вывода времени выполнения

### Пример использования

```
user@localhost:~$ gausscuda
Usage: gausscuda [-t] [-r] FILE...

Options:
  -r    filter radius (sigma)
  -t    print timings

Examples:
  gausscuda -t -r 0.1 sky.jpg face.bmp sun.bmp
  - Processing Gaussian blur on sky.jpg...
    Finished 0.02 sec
  - Processing Gaussian blur on face.bmp...
    Finished 0.014 sec
  - Processing Gaussian blur on sun.bmp...
    Finished 0.01 sec
```

