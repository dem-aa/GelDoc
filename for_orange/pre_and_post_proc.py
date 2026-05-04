import numpy as np
import pandas as pd
import cv2 
from pathlib import Path
import json
import subprocess
from collections import defaultdict

class PreProc:

    @staticmethod
    def load_img(image_path: Path):

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить: {image_path}")
        return image

    @staticmethod
    def rotate_image(image: np.ndarray, path_for_rotated: Path):

        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            cv2.imwrite(path_for_rotated, img)
            return img

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)

        angle = rect[2]
        angle = min([angle, 90 + angle], key=lambda x: abs(x))

        h, w = img.shape
        M = cv2.getRotationMatrix2D(rect[0], angle=angle, scale=1.0)
        rotated = cv2.warpAffine(img, M, (w, h))

        cv2.imwrite(path_for_rotated, rotated)
        print(f"Угол поворота для {path_for_rotated.name}: {angle:.4} Deg")

        return rotated
    
    @staticmethod
    def to_json(results: dict, json_dir: Path):
        
        with open(json_dir, 'w', encoding='UTF-8') as file:
            json.dump(results, file, indent=2)

    @staticmethod
    def make_image(results: dict, img: np.ndarray, image_path: Path):

        class_colors = {
            0: (255, 0, 0),    # Синий - zero
            1: (0, 255, 0),    # Зеленый - strip
            2: (0, 0, 255)     # Красный - line
        }

        img_with_preds = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        sx, sy = w / 1280, h / 1280

        for cls, bboxes in results.items():
            for (x1, y1, x2, y2, conf) in bboxes:
                x1, y1 = int(x1 * sx), int(y1 * sy) 
                x2, y2 = int(x2 * sx), int(y2 * sy)
                cv2.rectangle(img_with_preds, (x1, y1), (x2, y2), class_colors[cls], 1)

        cv2.imwrite(image_path, img_with_preds)

    @staticmethod
    def show_img(img_path: Path):

        subprocess.Popen(['feh', str(img_path)])



class PostProc:

    @staticmethod
    def convert_yolo_results(results, image_id, conf_threshold=0.1):
        """
        Конвертирует результаты YOLO в формат {image_id: {1: [(bbox, conf)], 2: [(bbox, conf)], 3: [(bbox, conf)]}}

        Args:
            results: результат model.predict() для одного изображения
            image_id: идентификатор изображения (может быть int или str)
            conf_threshold: порог уверенности (по умолчанию 0.1)

        Returns:
            dict: словарь в формате {image_id: {1: [(zeros, conf)], 2: [(strips, conf)], 3: [(lines, conf)]}}
        """

        # Инициализируем структуру для этого изображения
        grouped_data = {
            image_id: {
                1: [],  # zeros: список кортежей (bbox, conf)
                2: [],  # strips: список кортежей (bbox, conf)
                3: []   # lines: список кортежей (bbox, conf)
            }
        }

        # Получаем все bounding boxes
        boxes = results[0].boxes

        if len(boxes) == 0:
            print(f"Нет предсказаний для изображения {image_id}")
            return grouped_data

        # Обрабатываем каждый box
        for box in boxes:
            # Получаем координаты в формате xyxy
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Получаем уверенность
            conf = float(box.conf[0].cpu().numpy())

            # Фильтруем по уверенности
            if conf < conf_threshold:
                continue

            # Получаем класс
            cls = int(box.cls[0].cpu().numpy())

            # Конвертируем xyxy в xywh (x, y, width, height)
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)

            # Создаём bbox в формате [x, y, w, h]
            bbox = [x, y, w, h]

            # Создаём кортеж (bbox, conf)
            bbox_with_conf = (bbox, conf)

            # Добавляем в соответствующую категорию (класс +1, так как в вашем формате с 1)
            category_id = cls + 1  # 0->1, 1->2, 2->3

            if category_id in grouped_data[image_id]:
                grouped_data[image_id][category_id].append(bbox_with_conf)
            else:
                print(f"Предупреждение: неизвестная категория {category_id}")

        return grouped_data

    @staticmethod
    def group_yolo_results(data, image_ids=None):
        """Группирует zero, strips и lines в объекты по геометрическим условиям.

            Связывает элементы из разных категорий (zero, strips, lines) в единые
            группы на основе их пространственного расположения и пересечений.

            Args:
                data (dict): Входные данные в формате:
                    {
                        image_id (str/int): {
                            1: [(bbox, conf), ...],   # нулевые объекты (zeros)
                            2: [(bbox, conf), ...],   # полосы (strips)
                            3: [(bbox, conf), ...]    # линии (lines)
                        }
                    }
                image_ids (list, optional): Список ID изображений для обработки.
                    Если None, используются все ID из data. Defaults to None.

            Returns:
                dict: Результат группировки в формате:
                    {
                        image_id (str/int): [
                            {
                                "zero": (bbox, conf) or None,
                                "line": (bbox, conf) or None,
                                "strips": [(bbox, conf), ...] or None
                            },
                            ...
                        ]
                    }
        """

        def center(box):
            """Вычисляет центр bbox [x, y, w, h]"""
            x, y, w, h = box
            return x + w/2, y + h/2

        def is_inside_x(zero_box, line_box, tol=10):
            """Проверяет, попадает ли центр zero в горизонтальную проекцию line"""
            zx, zy, zw, zh = zero_box
            lx, ly, lw, lh = line_box

            zcx = zx + zw / 2

            return (lx - tol) <= zcx <= (lx + lw + tol)

        def is_below(zero_box, line_box):
            """Проверяет, находится ли line ниже zero"""
            _, zy, _, zh = zero_box
            _, ly, _, _ = line_box
            return ly > zy

        # Определяем список изображений для обработки
        if image_ids is None:
            image_ids = list(data.keys())

        UPD_data = {}

        for image_id in image_ids:
            if image_id not in data:
                print(f"Предупреждение: image_id {image_id} не найден в data")
                continue

            objects = data[image_id]

            # Извлекаем списки кортежей (bbox, conf)
            zeros_with_conf = objects.get(1, [])
            strips_with_conf = objects.get(2, [])
            lines_with_conf = objects.get(3, [])

            # Отслеживаем использованные элементы
            used_lines = set()
            used_zeros = set()

            # Извлекаем только bbox для геометрических вычислений
            zeros_boxes = [bbox for bbox, _ in zeros_with_conf]
            strips_boxes = [bbox for bbox, _ in strips_with_conf]
            lines_boxes = [bbox for bbox, _ in lines_with_conf]

            image_objects = []

            for zero_idx, (zero_box, zero_conf) in enumerate(zeros_with_conf):
                zx, zy, zw, zh = zero_box
                zcx, zcy = center(zero_box)

                best_line_idx = None
                best_line_box = None
                best_line_conf = None
                min_dy = float('inf')

                # Ищем подходящую line
                for line_idx, (line_box, line_conf) in enumerate(lines_with_conf):
                    # Пропускаем уже использованные line
                    if line_idx in used_lines:
                        continue

                    lx, ly, lw, lh = line_box
                    lcx, lcy = center(line_box)

                    # Проверяем условия
                    if not is_inside_x(zero_box, line_box):
                        continue
                    if not is_below(zero_box, line_box):
                        continue

                    dy = lcy - zcy
                    if dy < min_dy:
                        min_dy = dy
                        best_line_idx = line_idx
                        best_line_box = line_box
                        best_line_conf = line_conf

                # Ищем strips внутри найденной line
                matching_strips_with_conf = []

                if best_line_box is not None:
                    lx, ly, lw, lh = best_line_box

                    for strip_box, strip_conf in strips_with_conf:
                        sx, sy, sw, sh = strip_box
                        scx, scy = center(strip_box)

                        # Проверяем, находится ли strip внутри line
                        if (lx <= scx <= lx + lw and
                            ly <= scy <= ly + lh):
                            matching_strips_with_conf.append((strip_box, strip_conf))

                    # Помечаем line как использованную
                    used_lines.add(best_line_idx)

                # Формируем результат для текущего zero
                image_objects.append({
                    'zero': (zero_box, zero_conf),
                    'line': (best_line_box, best_line_conf) if best_line_box is not None else None,
                    'strips': matching_strips_with_conf
                })

                # Помечаем zero как использованный
                used_zeros.add(zero_idx)

            # Добавляем оставшиеся line, которые не были связаны ни с одним zero
            for line_idx, (line_box, line_conf) in enumerate(lines_with_conf):
                if line_idx not in used_lines:
                    image_objects.append({
                        'zero': None,
                        'line': (line_box, line_conf),
                        'strips': []
                    })

            UPD_data[image_id] = image_objects

        return UPD_data

    @staticmethod
    def convert_cls_results(cls_results, image_id):
        """Группирует zero, strips и lines в объекты по геометрическим условиям.

            Связывает элементы из разных категорий (zero, strips, lines) в единые
            группы на основе их пространственного расположения и пересечений.

            Args:
                cls_results (array): Входные данные в формате:
                    [
                        {
                            "lane_id"        : line coordinate from yolo,
                            "ml_y"           : strip coordinate from yolo,
                            "ml_conf"        : strip confidence from yolo,
                            "classical_y"    : strip coordinate from cls,
                            "classical_score": strip score from cls
                        },
                        ...
                    ]
                image_id: идентификатор изображения (может быть int или str)

            Returns:
                dict: Результат группировки в формате:
                    {
                        image_id (str/int): [
                            {
                                "line": [bbox],
                                "cls_y": [y_1, y_2, ...],
                                "cls_score": [score_1, score_2, ...]
                            },
                            ...
                        ]
                    }
        """

        cls_data = {}

        temp_cls_results = {
            'line': [],
            'cls_y': [],
            'cls_score': []
          }

        lines = []
        used_lines = set()

        for info in cls_results:
            for key, value in info.items():
                if key == 'lane_id':

                    if temp_cls_results['line'] == []:
                        temp_cls_results['line'] = value
                        used_lines.add(value)
                    elif temp_cls_results['line'] == value:
                        continue
                    elif temp_cls_results['line'] != value and value not in used_lines:
                        lines.append(temp_cls_results)
                        temp_cls_results = {
                        'line': value,
                        'cls_y': [],
                        'cls_score': []
                        }
                        used_lines.add(value)

                elif key == 'classical_y':
                    temp_cls_results['cls_y'].append(value)

                elif key == 'classical_score':
                    temp_cls_results['cls_score'].append(value)

        lines.append(temp_cls_results)

        cls_data[image_id] = lines

        return cls_data
    
    @staticmethod
    def create_dataframe(yolo_results, cls_results):
        """Создаёт DataFrame для Decider"""

        def create_empty_dataframe(check=False):
            """Создаёт пустой DataFrame"""

            yolo_columns = ['image_id', 'yolo_line_x', 'yolo_line_y', 'yolo_line_w', 'yolo_line_h', 'yolo_line_conf',
                            'yolo_strip_x', 'yolo_strip_y', 'yolo_strip_w', 'yolo_strip_h', 'yolo_strip_conf']

            cls_columns = ['image_id', 'yolo_line_x', 'yolo_line_y', 'yolo_line_w', 'yolo_line_h', 'cls_strip_center_y', 'cls_strip_center_y_score']

            # Объединяем все колонки и удаляем дубликаты
            all_columns = list(dict.fromkeys(yolo_columns + cls_columns))
            print(all_columns)

            # Создаем пустой DataFrame со всеми колонками
            df = pd.DataFrame(columns=all_columns)

            if check:
                print(f"data columns:\n{df.columns}")
                print(f"data shape: {df.shape}\n")

            return df

        def is_in_strip(strip, cls_strip_center_y):
            """Проверяет соответствие strip, обнаруженного классикой и yolo"""
            x, y, w, h = strip

            if y <= cls_strip_center_y <= y + h:
                return True
            return False

        # будем сохранять сюда строки будущего DataFrame
        rows_list = []

        for image_id, cls_data in cls_results.items():

            # if image_id != 1:
            #     continue

            print()
            print("=" * 50)
            print(f"IMAGE ID: {image_id}")
            print("=" * 50)
            print()

            # получаем все данные

            yolo_data = yolo_results[image_id]

            print(f"CLS_DATA: OK")
            print(f"YOLO_DATA: OK")

            # формируем строку в DataFrame

            # идём по всем найденным полоскам на изображении
            for line in cls_data:

                # если нет линии, то yolo тоже не нашла её, а т.к. мы пытаемся усреднять предсказания yolo и классики,
                # не имеет смысла рассматривать ситуацию, даже если линия есть в аннотациях, поэтому пропускаем
                # НО в будущем, надо будет обрабатывать даже те полосы. которые не нашла yolo
                if not line['line']:
                    continue

                yolo_line_x, yolo_line_y, yolo_line_w, yolo_line_h = line['line']

                # собираем предсказанные классикой стрипы в формат [(cls_y, cls_score), (cls_y, cls_score), ...]
                # и сортируем по возрастанию координаты
                cls_strips_with_scores = sorted(zip(line['cls_y'], line['cls_score']))

                # ищем набор стрипов, найденный yolo, к соответствующей line
                # а также уверенность yolo в этом line

                yolo_strips_with_confs = None
                yolo_line_conf = None
                for module in yolo_data:

                    # тут ситуация, когда yolo нашла zero, но не нашла подходящий line
                    if not module['line']:
                        continue

                    if tuple(module['line'][0]) == line['line']:
                        yolo_strips_with_confs = module['strips']
                        yolo_line_conf = module['line'][1]

                        break
                else:
                    print(f"SHIT IN YOLO!!!! не нашли соответствующую line для {line['line']}")


                print()
                print("=" * 50)
                print("LINE")
                print(f"yolo coordinates: {yolo_line_x, yolo_line_y, yolo_line_w, yolo_line_h}")
                print(f"yolo confidence: {yolo_line_conf}")
                print(f"amount of yolo strips: {len(yolo_strips_with_confs) if yolo_strips_with_confs else 0}")
                print(f"yolo strips with their confs: {yolo_strips_with_confs}")

                print()
                print(f"amount of cls strips: {len(cls_strips_with_scores)}")
                print(f"cls strips with their confs: {cls_strips_with_scores}")

                print("=" * 50)
                print("STRIPS")

                # окей, мы смогли сопоставить line и strips внутри них, теперь нужно сопоставить все стрипы и наконец собрать датасет

                # заведём множество, которое будет хранить все найденные strip, предсказанные yolo
                # она нужна, чтобы добавить в датасет те strip, которые нашла yolo, но не нашла классика
                yolo_strips_found = set()

                for cls_strip_center_y, cls_strip_center_y_score in cls_strips_with_scores:
                    # найдём соответствующий strip в предсказаниях yolo и аннотациях
                    # если где-то найти strip не удалось, то его (и его conf для yolo) принимаем его за None

                    # напишем функию, которая будет проверять, лежит ли предсказанная классикой точка внутри strip
                    # возможно, стоит взять допуск, но вообще говоря, весь смысл в том, что классика ТОЧНО найдёт все стрипы + ещё что-то

                    # ищем в предсказаниях yolo
                    yolo_strip_x, yolo_strip_y, yolo_strip_w, yolo_strip_h = [None, None, None, None]
                    yolo_strip_conf = None

                    print("=" * 30)
                    print("STRIP")
                    print()

                    if yolo_strips_with_confs:
                        for strip, conf in yolo_strips_with_confs:

                            if is_in_strip(strip, cls_strip_center_y):
                                yolo_strip_x, yolo_strip_y, yolo_strip_w, yolo_strip_h = strip
                                yolo_strip_conf = conf
                                yolo_strips_found.add((tuple(strip), conf))
                                break
                        else:
                            # сюда мы попадаем в случае, когда классика нашла стрипы, но не нашла соответствующие среди найденных yolo. Это значит либо:
                            # 1) yolo не нашла
                            # 2) yolo нашла, но классика не попала ни в один из них

                            print(f"Yolo didn't found strip, with center y coordinate {cls_strip_center_y}")

                    print()
                    print(f"yolo coordinates: {yolo_strip_x, yolo_strip_y, yolo_strip_w, yolo_strip_h}")
                    print(f"yolo confidence: {yolo_strip_conf}")

                    print()
                    print(f"cls center: {cls_strip_center_y}")
                    print(f"cls score: {cls_strip_center_y_score}")
                    print()

                    # предполагается, что тут мы нашли все интересующие нас значения и готовы формировать очередную строку DataFrame
                    rows_list.append({
                        'image_id': image_id,
                        'yolo_line_x': yolo_line_x,
                        'yolo_line_y': yolo_line_y,
                        'yolo_line_w': yolo_line_w,
                        'yolo_line_h': yolo_line_h,
                        'yolo_line_conf': yolo_line_conf,
                        'yolo_strip_x': yolo_strip_x,
                        'yolo_strip_y': yolo_strip_y,
                        'yolo_strip_w': yolo_strip_w,
                        'yolo_strip_h': yolo_strip_h,
                        'yolo_strip_conf': yolo_strip_conf,
                        'cls_strip_center_y': cls_strip_center_y,
                        'cls_strip_center_y_score' : cls_strip_center_y_score
                        })

                # Добавим в таблицу те strip, которые нашла yolo, но пропустила классика
                if yolo_strips_with_confs:
                    yolo_strips_set = {(tuple(strip), conf) for strip, conf in yolo_strips_with_confs}
                    yolo_found_cls_missed = yolo_strips_set - yolo_strips_found
                else:
                    yolo_found_cls_missed = set()

                print(f"YOLO found but CLS missed: {yolo_found_cls_missed}")

                # будем отталкиваться от strip, обнаруженных yolo

                if yolo_found_cls_missed:

                    for yolo_strip, conf in yolo_found_cls_missed:

                        yolo_x, yolo_y, yolo_w, yolo_h = yolo_strip
                        rows_list.append({
                            'image_id': image_id,
                            'yolo_line_x': yolo_line_x,
                            'yolo_line_y': yolo_line_y,
                            'yolo_line_w': yolo_line_w,
                            'yolo_line_h': yolo_line_h,
                            'yolo_line_conf': yolo_line_conf,
                            'yolo_strip_x': yolo_x,
                            'yolo_strip_y': yolo_y,
                            'yolo_strip_w': yolo_w,
                            'yolo_strip_h': yolo_h,
                            'yolo_strip_conf': conf,
                            'cls_strip_center_y': None,
                            'cls_strip_center_y_score' : None
                            })

        # после обработки всех изображений собираем DataFrame
        if rows_list:
            data = pd.DataFrame(rows_list)
        else:
            data = create_empty_dataframe()
        
        return data
