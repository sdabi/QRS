import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import defines

RED = (200, 50, 50)
GREEN = (0, 200, 100)
BLUE = (35, 100, 255)
GRAY = (150, 150, 150)
BLACK = (0, 0, 0)

def bold(text):
    return "\033[1m{}\033[0m".format(text)

def colored(color, text):
    return "\033[38;2;{};{};{}m{}\033[0m".format(color[0], color[1], color[2], bold(text))

def underline(text):
    return "\033[4m{}\033[0m".format(text)




def plot_embedded_vecs(vecs):
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(vecs)
    plt.scatter(vecs_2d[:,0], vecs_2d[:,1])
    for i, point in enumerate(vecs_2d):
        plt.annotate(i, (point[0], point[1]),
                     xytext=(point[0], point[1]+0.01), fontsize=8)
    plt.show()



def plot_HRK(hrk_list, titles_list):
    y_axis = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    for HRK, title in zip(hrk_list, titles_list):
        plt.plot(HRK, label=title)

    plt.ylim([0, 1.1])
    plt.yticks(y_axis)
    plt.xlim([1, 11])
    plt.legend()
    plt.grid()
    plt.show()

def print_colored_matrix(mat, colored_values=[], is_vec=0, all_positive=0, digits_after_point=1, print_col_index=0):
    space_for_print = (1-all_positive)+digits_after_point+3

    if print_col_index:
        print("     ", end="")
        for i in range(defines._NUM_OF_ITEMS):
            print(colored(GRAY, underline('{val:>{space_for_print}}'.format(val=i, space_for_print=space_for_print))), end="")

    r = -1
    # print("")  # new line
    # print(colored(GRAY, '{val:>{space_for_print}}'.format(val=r, space_for_print=3)), colored(GRAY, "|"), end="")

    for index_in_mat, val in np.ndenumerate(mat):
        print_in_black = 1
        if is_vec == 0 and r < index_in_mat[0]:
            print("")  # new line
            r = index_in_mat[0]
            print(colored(GRAY, '{val:>{space_for_print}}'.format(val=r, space_for_print=3)), colored(GRAY, "|"), end="")

        val = '{val:1.{digits_after_point}f}'.format(val=val, digits_after_point=digits_after_point)
        for index, color_list in enumerate(colored_values):
            # print("A", index_in_mat, index, color_list)
            if index_in_mat in color_list:
                # print("B", index_in_mat, index, color_list)
                print_in_black = 0
                if index == 0:
                    print(colored(RED, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="") #RED
                    break
                if index == 1:
                    print(colored(GREEN, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="") # GREEN
                    break
                if index == 2:
                    print(colored(BLUE, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="") #BLUE
                    break
        if print_in_black:
            print( '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print), end="")

    print("")
    return

def plot_cost_arrs(cost_arrs):
    cmap = plt.get_cmap('tab20')
    fig = plt.figure(figsize=(10, 20))
    ax = fig.add_subplot()

    x = range(len(cost_arrs[0]))
    for i, cost_arr in enumerate(cost_arrs):
        ax.plot(x, cost_arr, color=cmap(i), label=i, linewidth=2)
    ax.legend(range(len(cost_arrs)), loc='upper right')

    ymax = max(max(l) for l in cost_arrs)
    plt.ylim(ymin=0.0001,ymax=ymax)
    plt.ylabel('cost')
    plt.xlabel('iteration')

    plt.xlim(xmin=0,xmax=len(x)-1)
    plt.show()



def print_reco_matrix(mat, removed_inter_indicies):
    space_for_print = 3

    print("     ", end="")
    for i in range(defines._NUM_OF_ITEMS):
        print(colored(GRAY, underline('{val:>{space_for_print}}'.format(val=i, space_for_print=space_for_print))), end="")
    print("") # new line
    for row_num, vec in enumerate(mat):
        print(colored(GRAY, '{val:>{space_for_print}}'.format(val=row_num, space_for_print=3)), colored(GRAY, "|"), end="")
        for col_num, val in enumerate(vec):
            if val == 0:
                print('{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print),end="")
            if val == 1:
                print(colored(GREEN, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="")
            if val == -1:
                print(colored(RED, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="")
            if (row_num, col_num) in removed_inter_indicies:
                print(colored(BLUE, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="")
        print("") # new line
    return


