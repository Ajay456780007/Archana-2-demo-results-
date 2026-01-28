from graphviz import Digraph

def plot_anfis_architecture(n_inputs, n_rules):
    dot = Digraph()

    dot.node("X", f"Input\n({n_inputs})")
    dot.node("MF", "Gaussian MFs")
    dot.node("R", f"Rules\n({n_rules})")
    dot.node("N", "Normalize")
    dot.node("Z", f"Fuzzy Embedding\n({n_rules})")
    dot.node("D", f"Decoder\n({n_rules} → {n_inputs})")
    dot.node("L", "LightGBM")

    dot.edges([("X","MF"),("MF","R"),("R","N"),("N","Z")])
    dot.edge("Z","D", label="training")
    dot.edge("Z","L", label="classification")

    return dot

dot = plot_anfis_architecture(282,16)
dot.format = "png"          # or "pdf"
dot.render("anfis_architecture", view=True)

