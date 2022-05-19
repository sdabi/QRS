import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{}".format(r, g, b, text)


def plot_embedded_vecs(vecs):
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(vecs)
    plt.scatter(vecs_2d[:,0], vecs_2d[:,1])
    for i, point in enumerate(vecs_2d):
        plt.annotate(i, (point[0], point[1]))
    plt.show()



def plot_HRK(hrk_list, titles_list):
    y_axis = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    for HRK, title in zip(hrk_list, titles_list):
        plt.plot(HRK, label=title)

    plt.ylim([0, 1])
    plt.yticks(y_axis)
    plt.xlim([1, 11])
    plt.legend()
    plt.grid()
    plt.show()

def print_colored_matrix(mat, colored_values=[], is_vec=0, all_positive=0, digits_after_point=1):
    r = 0
    space_for_print = (1-all_positive)+digits_after_point+3
    for i, val in np.ndenumerate(mat):
        found = 0
        if is_vec == 0 and i[0] > r:
            print("")  # new line
            r = i[0]
        val = '{val:1.{digits_after_point}f}'.format(val=val, digits_after_point=digits_after_point)
        for index, color_list in enumerate(colored_values):
            if i in color_list:
                found = 1
                if index == 0:
                    print(colored(200, 50, 50, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="") #RED
                    break
                if index == 1:
                    print(colored(0, 200, 50, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="") # GREEN
                    break
                if index == 2:
                    print(colored(0, 165, 255, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="") #BLUE
                    break
        if found:
            continue
        print(colored(0, 0, 0, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="")
    print(colored(0, 0, 0, ""))  # new line
    return

def plot_cost_arrs(cost_arrs):
    x = np.arange(start=0, stop=(len(cost_arrs[0])), step=1)
    fig, ax = plt.subplots()
    for cost_arr in cost_arrs:
        ax.plot(x, cost_arr)
    ax.legend(range(len(cost_arrs)))
    ymax = max(max(l) for l in cost_arrs)
    plt.ylim(ymin=0.0001,ymax=ymax)
    plt.ylabel('cost')
    plt.xlabel('iteration')

    plt.xlim(xmin=0,xmax=len(x)-1)
    plt.subplots_adjust(top=0.8, bottom=0.4, left=0.10, right=0.5, hspace=0.4,
                        wspace=0.35)
    plt.show()


