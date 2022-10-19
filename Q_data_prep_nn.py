import defines
import numpy as np

def swap_mat_columns_by_permutation(matrix, permutation):
    orig_matrix = matrix.copy()
    matrix_to_ret = matrix.copy()
    for i in range(len(permutation)):
        matrix_to_ret[:,i] = orig_matrix[:,permutation[i]]
    return matrix_to_ret


def define_entangelment_score_graph():
    entangelment_score_graph = []
    for item1 in range(defines._NUM_OF_ITEMS):
        relations_vec = np.ones(defines._NUM_OF_ITEMS)
        for item2 in range(defines._NUM_OF_ITEMS):
             relations_vec[item2] /= 2**(bin(item1 ^ item2).count("1"))
        entangelment_score_graph.append(relations_vec)
    entangelment_score_graph = np.array(entangelment_score_graph)
    return entangelment_score_graph


def calc_items_relations_mat(matrix):
    items_relation_matrix = []
    for item in range(defines._NUM_OF_ITEMS):
        rows_idx_where_item_interacted = (matrix[:,item]==1) # get all rows index where item interacted
        rows_where_item_interacted = matrix[rows_idx_where_item_interacted,:] # get all row themselves
        if len(rows_where_item_interacted) > 0:
            rows_where_item_interacted = np.where(rows_where_item_interacted>0, 1, 0) # ignoring negetavie interactions
            count_of_common_items = sum(rows_where_item_interacted) # sum all the interactions
            #count_of_common_items = count_of_common_items/count_of_common_items[item] # normalizing
        else:
            count_of_common_items = np.zeros(defines._NUM_OF_ITEMS)
            count_of_common_items[item] = 1
        items_relation_matrix.append(count_of_common_items)
    items_relation_matrix = np.array(items_relation_matrix)
    items_relation_matrix += items_relation_matrix.T
    return items_relation_matrix


def calc_entangement_score(matrix, entangelment_score_graph):
    items_relations_mat = calc_items_relations_mat(matrix)
    #print("relation mat:\n", np.triu(items_relations_mat))
    items_relations_mat = (items_relations_mat**2)*100
    multiplication = np.multiply(items_relations_mat, entangelment_score_graph)
    score = np.triu(multiplication).sum()-np.trace(multiplication)
    #print(score)
    return score


def improve_entanglemet(matrix, max_iterations, recommendation_sets, set_to_min=0, imported_score_graph=[]):
    iter, best_ent_score = 0, 0
    best_ent_score = 0 if (set_to_min == 0) else 100000000


    if len(imported_score_graph)==0:
        entangelment_score_graph = define_entangelment_score_graph()
    else:
        print(imported_score_graph)
        entangelment_score_graph = imported_score_graph

    permutation = np.arange(defines._NUM_OF_ITEMS)
    best_permutation = permutation.copy()
    while True:
        mat_by_permutation = swap_mat_columns_by_permutation(matrix, permutation)
        ent_score = calc_entangement_score(mat_by_permutation, entangelment_score_graph)
        if iter == 0: print("basic permutation score:", ent_score)
        if (set_to_min == 0) and (ent_score > best_ent_score):
            best_ent_score = ent_score
            best_permutation = permutation.copy()
        if (set_to_min == 1) and (ent_score < best_ent_score):
            best_ent_score = ent_score
            best_permutation = permutation.copy()
            print("best score so far:", best_ent_score)

        if iter > max_iterations:
            break

        np.random.shuffle(permutation)
        iter += 1

    print("best permutation", best_permutation, "with score", best_ent_score)

    print("setting recommendation sets by new permutation")
    recommendation_sets_edited = []
    for rec_set in recommendation_sets:
        uninter_items = [np.where(best_permutation == item)[0][0] for item in rec_set[2]]
        recommendation_sets_edited.append((rec_set[0], np.where(best_permutation == rec_set[1])[0][0], uninter_items))

    return swap_mat_columns_by_permutation(matrix, best_permutation), recommendation_sets_edited


