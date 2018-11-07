from pomegranate import *

def crossintersection():
    
    stop = DiscreteDistribution({'True': 0.5, 'False': 0.5})
    car = DiscreteDistribution({'True': 0.5, 'False': 0.5})
    ped = DiscreteDistribution({'True': 0.5, 'False': 0.5})
    approach = DiscreteDistribution({'True': 0.5, 'False': 0.5})

    cross = ConditionalProbabilityTable(
        [['False','False','False','False','True', 0.05],
         ['False','False','False','False','False', 0.95],
         ['False','False','False','True', 'True', 0.6],
         ['False','False','False','True', 'False', 0.4],
         ['False','False','True','False', 'True', 0.05],
         ['False','False','True','False', 'False', 0.95],
         ['False','False','True','True', 'True', 0.9],
         ['False','False','True','True', 'False', 0.1],
         ['False','True','False','False', 'True', 0.05],
         ['False','True','False','False', 'False', 0.95],
         ['False','True','False','True', 'True', 0.4],
         ['False','True','False','True', 'False', 0.6],
         ['False','True','True','False', 'True', 0.05],
         ['False','True','True','False', 'False', 0.95],
         ['False','True','True','True', 'True', 0.55],
         ['False','True','True','True', 'False', 0.45],
         ['True','False','False','False', 'True', 0.05],
         ['True','False','False','False', 'False', 0.95],
         ['True','False','False','True', 'True', 0.85],
         ['True','False','False','True', 'False', 0.15],
         ['True','False','True','False', 'True', 0.05],
         ['True','False','True','False', 'False', 0.95],
         ['True','False','True','True', 'True', 0.99],
         ['True','False','True','True', 'False', 0.01],
         ['True','True','False','False', 'True', 0.05],
         ['True','True','False','False', 'False', 0.95],
         ['True','True','False','True', 'True', 0.85],
         ['True','True','False','True', 'False', 0.15],
         ['True','True','True','False', 'True', 0.05],
         ['True','True','True','False', 'False', 0.95],
         ['True','True','True','True', 'True', 0.9],
         ['True','True','True','True', 'False', 0.1]], 
         [stop, car, ped, approach])

    s0 = State(stop, name="stop")
    s1 = State(car, name="car")
    s2 = State(ped, name="ped")
    s3 = State(approach, name="approach")
    s4 = State(cross, name="cross")

    network = BayesianNetwork("intersection")
    network.add_nodes(s0, s1, s2, s3, s4)
    network.add_edge(s0, s4)
    network.add_edge(s1, s4)
    network.add_edge(s2, s4)
    network.add_edge(s3, s4)

    network.bake()
    return network