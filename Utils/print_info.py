def print_data_pair(data_pair):
    for single_data in data_pair:
        for key in single_data:
            print(key)
            print(single_data[key])
        break


def get_adjust_ratio(history_aw_dict, max_na_length, max_naw_length):

    na_node_adjust, naw_node_adjust, naw_aspect_adjust = 0, 0, 0
    total_node = len(history_aw_dict)
    total_na = 0
    for nid in history_aw_dict:
        total_na += len(history_aw_dict[nid])
        if len(history_aw_dict[nid]) > max_na_length:
            na_node_adjust += 1
        node_flag = 0
        for na in history_aw_dict[nid]:
            if len(history_aw_dict[nid][na]) > max_naw_length:
                naw_aspect_adjust += 1
                node_flag = 1
        if node_flag:
            naw_node_adjust += 1
    na_node_adjust_ratio = na_node_adjust / total_node
    naw_node_adjust_ratio = naw_node_adjust / total_node
    naw_aspect_adjust_ratio = naw_aspect_adjust / total_na

    adjust_ratio = (na_node_adjust_ratio, naw_node_adjust_ratio, naw_aspect_adjust_ratio)

    return adjust_ratio


def print_data_ratio(data_pair, mini_range, user_history_aw_dict, poi_history_aw_dict):

    max_up_length, max_ua_length, max_uaw_length, max_pa_length, max_paw_length, \
    max_pw_length, max_label_length, pos_neg_sample, top_number = mini_range

    user_adjust_ratio = get_adjust_ratio(user_history_aw_dict, max_ua_length, max_uaw_length)
    poi_adjust_ratio = get_adjust_ratio(poi_history_aw_dict, max_pa_length, max_paw_length)

    input_poi_adjust_number, input_user_adjust_number = 0, 0
    tag_adjust_number, total_tags_number, ugc_adjust_number = 0, 0, 0

    for single_data in data_pair:
        if len(single_data['src_Pw_seq']) > max_pw_length:
            input_poi_adjust_number += 1

        user_up = single_data['src_Up']
        if len(user_up) > max_up_length:
            input_user_adjust_number += 1

        total_tags_number += len(single_data['src_label_tags'])
        for aspect in single_data['src_label_tags']:
            if len(single_data['src_label_tags'][aspect]) > max_label_length:
                tag_adjust_number += 1
        if len(single_data['src_label_tags']) > top_number:
            ugc_adjust_number += 1

    print(" {:>17s}:\n".format("Adjust"),
          "{:>17s}: {:<4d} {:<17s} {:.2f}%\n".format("User_POI", max_up_length,
                                                     "limit user ratio:",
                                                     input_user_adjust_number / len(data_pair) * 100),
          "{:>17s}: {:<4d} {:<17s} {:.2f}%\n".format("User_Aspect", max_ua_length,
                                                     "limit user ratio:", user_adjust_ratio[0] * 100),
          "{:>17s}: {:<4d} {:<17s} {:.2f}%  {:<18s} {:.2f}%\n".format("User_Aspect_Word", max_uaw_length,
                                                                      "limit user ratio:", user_adjust_ratio[1] * 100,
                                                                      "limit aspect ratio:",
                                                                      user_adjust_ratio[2] * 100),
          "{:>17s}: {:<4d} {:<17s} {:.2f}%\n".format("POI_Aspect", max_pa_length,
                                                     "limit poi ratio:", poi_adjust_ratio[0] * 100),
          "{:>17s}: {:<4d} {:<17s} {:.2f}%  {:<18s} {:.2f}%\n".format("POI_Aspect_Word", max_paw_length,
                                                                      "limit poi ratio:", poi_adjust_ratio[1] * 100,
                                                                      "limit aspect ratio:", poi_adjust_ratio[2] * 100),
          '{:>17s}: {:<4d} {:<17s} {:.2f}%\n'.format("POI_Word_Length", max_pw_length,
                                                     "limit poi ratio:",
                                                     input_poi_adjust_number / len(data_pair) * 100),
          '{:>17s}: {:<4d} {:<17s} {:.2f}%  {:<18s} {:.2f}%\n'.format("Label_length", max_label_length,
                                                                      "limit tag ratio:",
                                                                      tag_adjust_number / total_tags_number * 100,
                                                                      "limit ugc ratio:",
                                                                      ugc_adjust_number / len(data_pair) * 100)
          )