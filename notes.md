Вот так надо делать:

1. Сделать абстрактную фабрику
2. Реализовать интерфейс в этой фабрике для чтения файла(всевозможные расширения запихать в MAP)

3. Реализовать Абстрактный класс IReader с интерфейсом read и его потом переопределять в дочерних классах
4. Наследоваться от IReader всякими JPGReader, PNGReader
5. Так же реализовать методы для записи в файл

6. Чтение файла и запись в файл картинки нагуглить
