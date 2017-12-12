from sklearn import tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pds
from sklearn.externals.six import StringIO
import numpy as np
import pydotplus

if __name__ == '__main__':
    with open('lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    #print(lenses_dict)
    lenses_pds = pds.DataFrame(lenses_dict)
    print(lenses_pds)
    le = LabelEncoder()
    for col in lenses_pds.columns:
        lenses_pds[col] = le.fit_transform(lenses_pds[col])
    print(lenses_pds)

    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(lenses_pds.values.tolist(), lenses_target)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file = dot_data,
                         feature_names = lenses_pds.keys(),
                         class_names = clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree.pdf')

    print(clf.predict([[1,1,1,0]]))
