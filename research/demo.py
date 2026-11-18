from .engine import Evidence, Criterion, assess

if __name__ == "__main__":
    evidence = [Evidence(page=2, text="Prototype tested in a laboratory environment.", target="Prototype A", source_id="demo")]
    criteria = [Criterion("lab", 4, "Prototype tested in a laboratory environment")]
    print(assess(evidence, criteria, "Prototype A"))
