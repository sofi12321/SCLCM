from collections import defaultdict
import random

def create_batches(batch_size, group_labels, kk = 3):
    # Группируем датапоинты по их номерам групп
    group_to_points = defaultdict(list)
    for idx, group in enumerate(group_labels):
        group_to_points[group].append(idx)

    # Преобразуем группы в список и перемешиваем их
    groups = list(group_to_points.keys())
    random.shuffle(groups)  # Перемешиваем группы
    for group in groups:
        random.shuffle(group_to_points[group])

    batches = []

    # Пока есть группы с более чем одним датапоинтом
    while groups:
        batch = []
        used_groups_in_batch = set()  # Множество для отслеживания групп в текущем батче

        # Check the rest is for one batch
        if sum([len(v) for k, v in group_to_points.items() if k in groups]) <= batch_size:
            for group in groups.copy():
                batch += group_to_points[group]
                used_groups_in_batch.add(group)

            groups = []
        random.shuffle(groups)
        # Пытаемся добавить группы в батч
        for group in groups.copy():
            # Если в группе есть как минимум 2 датапоинта и в батче есть место
            if (len(group_to_points[group]) >= 2) and ((len(batch) + 2 <= batch_size) or (len(batch) == 0)):
                # Добавляем 2 или более датапоинтов из группы
                if len(group_to_points[group]) > 2:
                    n = max(3, min(len(group_to_points[group]), batch_size // kk))
                    num_points_to_add = np.random.randint(2, n)
                    if len(group_to_points[group]) - num_points_to_add < 2:
                        num_points_to_add = len(group_to_points[group])
                    if (num_points_to_add + len(batch) > batch_size) and (len(batch) > 0):
                        num_points_to_add = batch_size - len(batch)
                    if num_points_to_add < 2:
                        num_points_to_add = 2

                elif len(group_to_points[group]) == 2:
                    num_points_to_add = 2
                else:
                    num_points_to_add = 0

                batch.extend(group_to_points[group][:num_points_to_add])
                used_groups_in_batch.add(group)  # Добавляем группу в использованные
                group_to_points[group] = group_to_points[group][num_points_to_add:]  # Удаляем добавленные датапоинты
                # Если в группе осталось меньше 2 датапоинтов, удаляем её из списка групп
                if len(group_to_points[group]) < 2:
                    groups.remove(group)
                # if len(batch) >= batch_size:
                #     break
            # Если батч достиг размера или содержит 2 разные группы, выходим из цикла
            if len(batch) >= batch_size:
                break

        # Если батч содержит как минимум 2 датапоинта и 2 разные группы, добавляем его в список батчей
        if len(batch) >= 2 and len(used_groups_in_batch) >= 2:
            batches.append(batch)
        else:
            # Если больше нельзя сформировать батчи, выходим
            break

    return batches

# Пример использования
# batch_size = 64
# group_labels = [1, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 5, 5]

# batches = create_batches(batch_size, train_pids)
# print(batches)

# for i in batches:
#     print(len(i), len(np.unique(train_pids[i])))
