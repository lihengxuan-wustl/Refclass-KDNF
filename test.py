import anchor_tabular

explainer=anchor_tabular.AnchorTabularExplainer(None,None,None,None,None)
explainer.m=6
sets=[[1,2,3],[1,4,5],[2,3],[1,6],[3,5,6],[1,4,6]]
weights=[3,4,5,6,3,4]
print(explainer.partial_greedy(sets,weights,1,1))