import numpy as np

def fd(distance, threshold):
    return -distance + threshold

def calc_reward(score, distance):
    reward = score * fd(distance, threshold=4)
    return reward

def global_score(score_table):
    flat_indices = np.argsort(score_table, axis=None)[::-1]
    sorted_indices = np.unravel_index(flat_indices, score_table.shape)
    sorted_indices_list = list(zip(sorted_indices[0], sorted_indices[1]))

    counter = 0
    largest_score, largest_index = 0, [0, 0]
    localizability_euclid = 0
    localizability_manhattan = 0

    for index in sorted_indices_list:
        if counter == 0: # largest score
            largest_score = score_table[index[0], index[1]]
            largest_index[0], largest_index[1] = index[0], index[1]
            localizability_euclid += calc_reward(largest_score, 0)
            localizability_manhattan += calc_reward(largest_score, 0)
        else:
            point_score = score_table[index[0], index[1]]

            # ユークリッド距離
            dist_x, dist_y = largest_index[0] - index[0], largest_index[1] - index[1]
            dist_euclud = (dist_x ** 2 + dist_y ** 2) ** 0.5

            # マンハッタン距離
            dist_xm, dist_ym = abs(largest_index[0] - index[0]), abs(largest_index[1] - index[1])
            dist_m = dist_xm + dist_ym
            reward_euclid = calc_reward(point_score, dist_euclud)
            reward_manhattan = calc_reward(point_score, dist_m)
            localizability_euclid += reward_euclid
            localizability_manhattan += reward_manhattan
        
        counter += 1

    return localizability_euclid, localizability_manhattan