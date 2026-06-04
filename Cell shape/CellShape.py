import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.measure import regionprops

def classify_shape(prop):
    """Определяет форму и метрики одной клетки"""
    area = prop.area
    perimeter = prop.perimeter

    # Кстати вот и диаметры
    minor_axis = prop.minor_axis_length
    major_axis = prop.major_axis_length

    # Защита от артефактов
    if perimeter == 0 or minor_axis == 0:
        return "unknown", 0.0, 0.0


    # ---------- Характеристики ----------

    # Отношение площади к перемитру ( -> 1 = круг)
    circularity = (4 * math.pi * area) / (perimeter ** 2)

    # Вытянутость ( -> 1 = вытянутая, -> 0 = круг)
    eccentricity = prop.eccentricity

    # Отношение площади к конвексу ( -> 0 = ветвистая)
    solidity = prop.solidity

    # Отношение сторон
    aspect_ratio = major_axis / minor_axis

    # Можно еще поугадывать параметры но вроде так терпимо
    if eccentricity > 0.85 and aspect_ratio > 3:
        shape = "spindle-shaped"
    elif solidity < 0.85:
        shape = "multi-branched"
    elif circularity > 0.85:
        shape = "round"
    else:
        shape = "polygonal"

    # Возвращаем форму (и все метрики для логов)
    return shape, circularity, aspect_ratio, eccentricity, solidity


def process_masks(masks):
    """Обработка маски в отдельные клетки"""
    props = regionprops(masks)
    cell_data = {}

    # Защита от артефактов
    for prop in props:
        if prop.area < 50:
            continue

        shape, circ, aspect, ecc, sol = classify_shape(prop)

        cell_data[prop.label] = {
            "shape": shape,
            "circularity": round(circ, 3),
            "aspect_ratio": round(aspect, 3),
            "eccentricity": round(ecc, 3),
            "solidity": round(sol, 3),
            "centroid": prop.centroid
        }

    # Возвращается инфа по каждой отдельной клетке
    return cell_data

def visual(image_path, mask_npy_path, logs):
    # Исходная картинка
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Нет картинки {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_img = img_rgb.copy()

    # Маска
    masks = np.load(mask_npy_path)

    # Дата по формам
    cell_data = process_masks(masks)

    shape_counts = {"polygonal": 0, "round": 0, "multi-branched": 0, "spindle-shaped": 0, "unknown": 0}

    # Словарь с цветами (RGB) для каждой формы
    shape_colors = {
        "round": (255, 0, 0),          # Красный
        "spindle-shaped": (0, 255, 0), # Зеленый
        "multi-branched": (0, 0, 255), # Синий
        "polygonal": (255, 255, 0),    # Желтый
        "unknown": (255, 255, 255)     # Белый
    }

    if logs:
      # Логи в консольку
      print(f"{'ID':<4} | {'Ecc':<6} | {'Sol':<6} | {'Circ':<6} | {'Asp':<6} | {'Shape'}")
      print("-" * 70)

    for cell_id, info in sorted(cell_data.items()):
        # Логи в консольку
        if logs:
          print("-" * 70)
          print(f"{cell_id:<4} | {info['eccentricity']:<6} | {info['solidity']:<6} | "
                f"{info['circularity']:<6} | {info['aspect_ratio']:<6} | {info['shape']}")

        # Статистика по форме
        shape = info["shape"]
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

        # Отрисовка маски
        single_cell_mask = (masks == cell_id).astype(np.uint8)
        contours, _ = cv2.findContours(single_cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Получаем нужный цвет для текущей формы (если формы нет в словаре, будет белый)
        color = shape_colors.get(shape, (255, 255, 255))
        cv2.drawContours(res_img, contours, -1, color, 2)

        # # ID на рисунке
        # y0, x0 = info["centroid"]
        # cv2.putText(res_img, str(cell_id), (int(x0) - 8, int(y0) + 4),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # Итог по дате
    print("ИТОГ:")
    for s, c in shape_counts.items():
        if c > 0:
            print(f"  - {s}: {c} шт.")
    print(f"Всего клеток: {len(cell_data)}")
    print("""
    "round": "Красный",
    "spindle-shaped": "Зеленый",
    "multi-branched": "Синий",
    "polygonal": "Желтый"
    "unknown": "Белый"
    """)

    # Отрисовка
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(res_img)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p = 62  # <- менять для изменения фото

    IMAGE_FILE = f"mask_for_shape/{p}/orig_0.png"
    MASK_FILE = f"mask_for_shape/{p}/mask_0.npy"

    # `True` для детальной инфы по каждой клетке
    #  сюда же можно выводить диаметры но мне уже лень
    LOGS = False

    visual(IMAGE_FILE, MASK_FILE, LOGS)