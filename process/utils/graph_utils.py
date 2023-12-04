

def get_full_connected_edge(box_object_list):
    full_connected_edge_list = []
    for box_object1 in box_object_list:
        for box_object2 in box_object_list:
            if box_object1 == box_object2:
                continue
            full_connected_edge_list.append((box_object1, box_object2))
    return full_connected_edge_list