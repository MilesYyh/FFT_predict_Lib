from graphviz import Graph

f = Graph('finite_state_machine', filename='fsm.gv')
f.attr(rankdir='LR', size='8,5')

f.attr('node', shape='doublecircle')
f.node('ILE-60')
f.attr('node', shape='circle')
f.node('VAL-40')
f.node('MET-42')
f.node('ILE-61')
f.node('THR-73')

#por puente de hidrógeno
f.edge('ILE-60', 'VAL-40', label='0.722')
f.edge('ILE-60', 'MET-42', label='0.533')
f.edge('ILE-61', 'THR-73', label='0.780')

#add distances
f.edge('ILE-60', 'ILE-61', label='5.9', style="dashed")
f.edge('ILE-60', 'THR-73', label='5.4', style="dashed")

f.edge('VAL-40', 'MET-42', label='6.6', style="dashed")
f.edge('VAL-40', 'ILE-61', label='9.6', style="dashed")
f.edge('VAL-40', 'THR-73', label='10.4', style="dashed")

f.edge('MET-42', 'ILE-61', label='6.2', style="dashed")
f.edge('MET-42', 'THR-73', label='10.7', style="dashed")

f.view()