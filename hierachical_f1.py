from collections import defaultdict

class DAG:
    
    def __init__(self):
        self.graph = {}

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def add_edge(self, parent, children):
        
        for child in children:
            if parent in self.graph and child in self.graph:
                self.graph[parent].append(child)
            else:
                raise ValueError("Nodes not in graph")

    def get_ancestors(self, node):
        ancestors = set()

        def dfs(curr_node):
            for parent, children in self.graph.items():
                if curr_node in children:
                    ancestors.add(parent)
                    dfs(parent)

        dfs(node)
        return ancestors



   
dag=DAG()
dag.add_node("Persuasion")
dag.add_node("Ethos")
dag.add_node("Pathos")
dag.add_node("Logos")

dag.add_edge("Persuasion",["Ethos","Pathos","Logos"])

dag.add_node("Ad Hominem")
dag.add_node("Justification")
dag.add_node("Reasoning")

dag.add_edge("Ethos",["Ad Hominem"])
dag.add_edge("Logos",["Justification","Reasoning"])

dag.add_node("Name calling/Labeling")
dag.add_node("Doubt")
dag.add_node("Smears")
dag.add_node("Reductio ad hitlerum")

dag.add_edge("Ad Hominem",["Name calling/Labeling","Doubt","Smears","Reductio ad hitlerum"])

dag.add_node("Bandwagon")
dag.add_node("Appeal to authority")
dag.add_node("Glittering generalities (Virtue)")

dag.add_edge("Ethos",["Bandwagon","Appeal to authority","Glittering generalities (Virtue)"])
dag.add_edge("Justification",["Bandwagon","Appeal to authority"])

dag.add_node("Appeal to (Strong) Emotions")
dag.add_node("Exaggeration/Minimisation")
dag.add_node("Loaded Language")
dag.add_node("Flag-waving")
dag.add_node("Appeal to fear/prejudice")
dag.add_node("Transfer")

dag.add_edge("Pathos",["Appeal to (Strong) Emotions","Exaggeration/Minimisation","Loaded Language","Flag-waving","Appeal to fear/prejudice","Transfer"])
dag.add_edge("Justification",["Flag-waving","Appeal to fear/prejudice"])

dag.add_node("Slogans")

dag.add_edge("Justification",["Slogans"])

dag.add_node("Repetition")
dag.add_node("Obfuscation, Intentional vagueness, Confusion")

dag.add_edge("Logos",["Repetition","Obfuscation, Intentional vagueness, Confusion"])

dag.add_node("Distraction")
dag.add_node("Simplification")
dag.add_edge("Reasoning",["Distraction","Simplification"])

dag.add_node("Misrepresentation of Someone's Position (Straw Man)")
dag.add_node("Presenting Irrelevant Data (Red Herring)")
dag.add_node("Whataboutism")
dag.add_node("Causal Oversimplification")
dag.add_node("Black-and-white Fallacy/Dictatorship")
dag.add_node("Thought-terminating cliché")

dag.add_edge("Distraction",["Misrepresentation of Someone's Position (Straw Man)","Presenting Irrelevant Data (Red Herring)","Whataboutism"])
dag.add_edge("Simplification",["Causal Oversimplification","Black-and-white Fallacy/Dictatorship","Thought-terminating cliché"])

dag.add_edge("Ad Hominem",["Whataboutism"])
