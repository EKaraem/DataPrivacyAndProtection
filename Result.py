import matplotlib.pyplot  as plt
import numpy as np
import collections
import networkx as nx
import csv
import sys
import os

try:  # Python 3.x
    import urllib.request as urllib
except ImportError:  # Python 2.x
    import urllib
import io
import zipfile

from numpy.core._multiarray_umath import ndarray


def c_merge(d , d1 , k):
    res = d1 - d[ k ] + compute_I(d[ k + 1: min(len(d) , 2 * k) ])
    return res


def c_new(d , k):
    t = d[ k:min(len(d) , 2 * k - 1) ]
    res = compute_I(t)
    return res


def compute_I(d):
    d_i = d[ 0 ]
    res = 0
    for d_j in d:
        res += d_i - d_j
    return res


def dp_graph_anonymization_optimzation(array_degrees , k , pos_init):
    # completing . . . .
    temp_memo = [ ]
    temp_cost = [ ]
    lst = [ ]
    for i in range(pos_init , len(array_degrees)):
        if i < ((2 * k) - 1):
            a = [ array_degrees[ 0 ] ] * (i + 1)
            cost1.append(compute_I(array_degrees[ 0:i + 1 ]))
            memo1.append(a)
        elif i >= ((2 * k) - 1):
            temp_memo.clear()
            temp_cost.clear()
            min_t = max(k , (i + 1) - 2 * k + 1)
            max_t = (i + 1) - k
            for t in range(min_t , max_t + 1):
                zero_t = array_degrees[ 0:t ]
                t_i = array_degrees[ t:i + 1 ]
                for l in range(0 , len(memo1)):
                    if len(memo1[ l ]) == len(zero_t):
                        t_i = [ array_degrees[ t ] ] * (i - t + 1)
                        zero_t = memo1[ l ]
                        arr = zero_t + t_i
                        temp_memo.append(arr)
                        c = compute_I(array_degrees[ t:i + 1 ])
                        temp_cost.append(cost1[ l ] + c)
                        break
            id = temp_cost.index(min(temp_cost))
            memo1.append(temp_memo[ id ])
            cost1.append(min(temp_cost))

    lent = len(memo1) - 1
    return (memo1[ lent ])


def dp_graph_anonymization(array_degrees , k , pos_init):
    # completing . . . .
    temp_memo = [ ]
    temp_cost = [ ]
    lst = [ ]
    # print("arraydegree{} ".format(array_degrees))
    for i in range(pos_init , len(array_degrees)):
        if i < ((2 * k) - 1):
            a = [ array_degrees[ 0 ] ] * (i + 1)
            cost.append(compute_I(array_degrees[ 0:i + 1 ]))
            memo.append(a)
        elif i >= ((2 * k) - 1):
            temp_memo.clear()
            temp_cost.clear()
            # min_t = max(k , (i + 1) - 2 * k + 1)
            min_t = k
            max_t = (i + 1) - k
            for t in range(min_t , max_t + 1):
                zero_t = array_degrees[ 0:t ]
                t_i = array_degrees[ t:i + 1 ]
                for l in range(0 , len(memo)):
                    if len(memo[ l ]) == len(zero_t):
                        t_i = [ array_degrees[ t ] ] * (i - t + 1)
                        zero_t = memo[ l ]
                        arr = zero_t + t_i
                        temp_memo.append(arr)
                        c = compute_I(array_degrees[ t:i + 1 ])
                        temp_cost.append(cost[ l ] + c)
                        break
            zero_i = array_degrees[ 0:i + 1 ]
            zero_i_cost = compute_I(zero_i)
            if min(temp_cost) < zero_i_cost:
                id = temp_cost.index(min(temp_cost))
                memo.append(temp_memo[ id ])
                cost.append(min(temp_cost))
            else:
                zero_i = [ array_degrees[ 0 ] ] * (i + 1)
                memo.append(zero_i)
                cost.append(zero_i_cost)

    lent = len(memo) - 1
    return (memo[ lent ])


def greedy_rec_algorithm(array_degres , k_degree , pos_init , extension):
    # complete this function
    if pos_init + extension >= len(array_degres) - 1:
        for i in range(pos_init , len(array_degres)):
            array_degres[ i ] = array_degres[ pos_init ]
            return array_degres
    else:
        d1 = array_degres[ pos_init ]
        c_merge_cost = c_merge(array_degres , d1 , pos_init + extension)
        c_new_cost = c_new(d , pos_init + extension)

        if c_merge_cost > c_new_cost:
            for i in range(pos_init , pos_init + extension):
                array_degres[ i ] = d1
            greedy_rec_algorithm(array_degres , k_degree , pos_init + extension , k_degree)
        else:
            greedy_rec_algorithm(array_degres , k_degree , pos_init , extension + 1)


def construct_graph(tab_index , anonymized_degree):
    graph = nx.Graph()
    if sum(anonymized_degree) % 2 == 1:
        return 0

    while True:
        if not all(di >= 0 for di in anonymized_degree):
            return -1
        if all(di == 0 for di in anonymized_degree):
            return graph
        v = np.random.choice((np.where(np.array(anonymized_degree) > 0))[ 0 ])
        dv = anonymized_degree[ v ]
        anonymized_degree[ v ] = 0
        for index in np.argsort(anonymized_degree)[ -dv: ][ ::-1 ]:
            if index == v:
                return -2
            if not graph.has_edge(tab_index[ v ] , tab_index[ index ]):
                graph.add_edge(tab_index[ v ] , tab_index[ index ])
                anonymized_degree[ index ] = anonymized_degree[ index ] - 1


if __name__ == "__main__":
    cost = [ ]
    memo = [ ]
    memo1 = [ ]
    cost1 = [ ]
    G = nx.Graph()
    # -----------

    # with open('Dataset/quakers_nodelist.csv' , 'r') as nodecsv:  # Open the file
    #     nodereader = csv.reader(nodecsv)  # Read the csv
    #     # Retrieve the data (using Python list comprhension and list slicing to remove the header row, see footnote 3)
    #     nodes = [ n for n in nodereader ][ 1: ]
    #
    # node_names = [ n[ 0 ] for n in nodes ]  # Get a list of only the node names
    #
    # with open('Dataset/quakers_edgelist.csv' , 'r') as edgecsv:  # Open the file
    #     edgereader = csv.reader(edgecsv)  # Read the csv
    #     edges = [ tuple(e) for e in edgereader ][ 1: ]  # Retrieve the data
    # G.add_nodes_from(node_names)
    # G.add_edges_from(edges)
    url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

    sock = urllib.urlopen(url)  # open URL
    s = io.BytesIO(sock.read())  # read into BytesIO "file"
    sock.close()

    zf = zipfile.ZipFile(s)  # zipfile object
    txt = zf.read('football.txt').decode()  # read info file
    gml = zf.read('football.gml').decode()  # read gml data
    # throw away bogus first line with # from mejn files
    gml = gml.split('\n')[ 1: ]
    G = nx.parse_gml(gml)
    # parse gml data

    # print(txt)
    # print degree for each team - number of games
    # for n , d in G.degree():
    #     print('%s %d' % (n , d))

    # options = {
    #     'node_color': 'black' ,
    #     'node_size': 50 ,
    #     'line_color': 'grey' ,
    #     'linewidths': 0 ,
    #     'width': 0.1 ,
    # }
    # nx.draw(G , **options)
    # plt.show()

    print("Origanl Grap  :{}".format(nx.info(G)))
    print("node Origanl Grap  :{}".format(G.nodes()))
    print("Edges Origanl Grap  :{}".format(G.edges()))
    # print("average shortest path length Origanl Grap  :{}".format(nx.average_shortest_path_length(G)))
    print("Origanl Grap average_clusterin :{}".format(nx.average_clustering(G)))

    # plt.suptitle('Orignal  Graph')
    # # nx.draw(G , with_labels=True)
    # nx.draw(G)
    # plt.margins(x=-0.3 , y=-0.4)
    # plt.show()
    # Degree arrays preparation
    d = [ x[ 1 ] for x in G.degree() ]
    d1 = d
    g = dp = o = 0
    array_index = np.argsort(d)[ ::-1 ]
    array_degrees = np.sort(d)[ ::-1 ]
    arr_d = np.sort(d1)[ ::-1 ]
    R = [ ]
    R1 = [ ]
    k = [ ]
    k_gre = [ ]
    k_dp = [ ]
    k_o = [ ]
    print("Orignail Degree: \n")
    print(arr_d)
    for k_degree in range(3 , 21):
        g = dp = o = 0

        print("K : {}".format(k_degree))
        # Dynamic Programming
        array_degrees_DP = dp_graph_anonymization(array_degrees , k_degree , 0)
        # Dynamic Programming Op
        array_degrees_DP1 = dp_graph_anonymization_optimzation(array_degrees , k_degree , 0)
        # greedy
        greedy_rec_algorithm(array_degrees , k_degree , 0 , k_degree)
        print("Dynamic programming: \n")
        print(array_degrees_DP)
        print("Dynamic programming OP: \n")
        print(array_degrees_DP1)
        print("Greedy: \n")
        print(array_degrees)
        sum_g = sum_d = sum_dO = 0
        # print(" LENHT ")
        # print(len(arr_d))
        # for j in range(0 , len(arr_d)):
        #     sum_g += array_degrees[ j ] - arr_d[ j ]
        #     sum_d += array_degrees_DP[ j ] - arr_d[ j ]
        #     sum_dO += array_degrees_DP1[ j ] - arr_d[ j ]
        sum_g=sum(array_degrees)
        sum_d=sum(array_degrees_DP)
        sum_dO=sum(array_degrees_DP1)
        sum_ar=sum(arr_d)

        if sum_g != 0:
            sd_greedy = sum_g / 2
        if sum_d != 0:
            sd_DP = sum_d / 2
        if sum_dO != 0:
            sd_DP1 = sum_dO / 2
        if sum_g != 0 and sum_d != 0:
            R.append((sum_g-sum_ar)/(sum_d-sum_ar))
        if sum_dO != 0 and sum_g != 0:
            R1.append((sum_g-sum_ar)/(sum_dO-sum_ar))
        print(R)
        print(R1)
        # -----------
        graph_greedy = construct_graph(array_index , array_degrees)
        if graph_greedy not in [ 0 , -1 , -2 ]:
            k_gre.append(k_degree)
            g = 1
            print("-------------------")
            print("Greedy Anonyimzation Grap :{}".format(nx.info(graph_greedy)))
            print("node Greedy Grap  :{}".format(graph_greedy.nodes()))
            print("Edges Greedy Grap  :{}".format(graph_greedy.edges()))
            # print("average shortest path length Greedy Anonyimzation Grap  :{}".format(nx.average_shortest_path_length(graph_greedy)))
            print("Greedy Anonyimzation Grap average clustering  :{}".format(nx.average_clustering(graph_greedy)))
            print("Greedy Anonyimzation Grap Structural difference  :{}".format(sd_greedy))
        else:
            print("Not Graph")
        graph_DP = construct_graph(array_index , array_degrees_DP)
        if graph_DP not in [ 0 , -1 , -2 ]:
            k_dp.append(k_degree)
            dp = 1
            print("-------------------")
            print("DP Anonyimzation Grap:{}".format(nx.info(graph_DP)))
            print("node DP Grap  :{}".format(graph_DP.nodes()))
            print("Edges DP Grap  :{}".format(graph_DP.edges()))
            # print("average shortest path length DP Anonyimzation Grap  :{}".format(nx.average_shortest_path_length(graph_DP)))
            print("DP Anonyimzation Grap average_clusterin :{}".format(nx.average_clustering(graph_DP)))
            print("DP Anonyimzation Grap Structural difference  :{}".format(sd_DP))
        else:
            print("Not Grap")
        graph_DP1 = construct_graph(array_index , array_degrees_DP1)
        if graph_DP1 not in [ 0 , -1 , -2 ]:
            k_o.append(k_degree)
            o = 1
            print("-------------------")
            print("DP_O Anonyimzation Grap:{}".format(nx.info(graph_DP1)))
            print("node DP_O Grap  :{}".format(graph_DP1.nodes()))
            print("Edges DP_O Grap  :{}".format(graph_DP1.edges()))
            # print("average shortest path length DP Anonyimzation Grap  :{}".format(nx.average_shortest_path_length(graph_DP)))
            print("DP_O Anonyimzation Grap average_clusterin :{}".format(nx.average_clustering(graph_DP1)))
            print("DP_O Anonyimzation Grap Structural difference  :{}".format(sd_DP1))
        else:
            print("Not Grap")
        if (g == 1) and (dp == 1) and (o == 1):
            k.append(k_degree)

        d = [ x[ 1 ] for x in G.degree() ]
        array_index = np.argsort(d)[ ::-1 ]
        array_degrees = np.sort(d)[ ::-1 ]
        arr_d = np.sort(d1)[ ::-1 ]
        g = dp = o = 0

    print("R :{}".format(R))
    print("R :{}".format(len(R)))
    print("R1 :{}".format(R1))
    print("R1 :{}".format(len(R1)))
    print("k_degree :{}".format(k))
    print("k_degree :{}".format(len(k)))

    # if len(k)!=0:
    #     labels = k
    #
    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars
    #
    # fig , ax = plt.subplots()
    # rects1 = ax.bar(x - width / 2 , R1 , width , label='DP_O')
    # rects2 = ax.bar(x + width / 2 , R , width , label='DP')
    #
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Scores')
    # ax.set_title('Scores by group and gender')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.legend()

    # plt.show()
# ----------------------------------
