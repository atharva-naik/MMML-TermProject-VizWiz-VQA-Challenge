# plot venn diagram according to skill labels.
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# PREDS_PATH: str = "./experiments/frozen_vilt2/formatted_pred.json"
def plot_vizwiz_skill_sets_venn_diagram(path: str="./data/skill/vizwiz_skill_typ_val.csv"):
    img_to_skill = {}
    data = pd.read_csv(path).to_dict("records")
    TXT = set()
    OBJ = set()
    COL = set()
    # CNT = set()
    for i, item in enumerate(data):
        if item["TXT"]>2: TXT.add(i)
        if item["OBJ"]>2: OBJ.add(i)
        if item["COL"]>2: COL.add(i)
        # if item["CNT"]>2: CNT.add(i)
    venn3([TXT, OBJ, COL], set_colors=('blue', 'red', 'green'))
    all = set()
    all = all.union(TXT)
    all = all.union(OBJ)
    all = all.union(COL)
    print(len(all))
    plt.text(-0.35, -0.15, 'TXT')
    plt.text(0.35, -0.15, 'COL')
    plt.text(-0.5, 0.4, 'OBJ')
    # Add labels and title
    plt.title("Overlap between TXT, OBJ and COL skills")
    plt.savefig("./skills_overlap_venn_diagram")

# main
if __name__ == "__main__":
    plot_vizwiz_skill_sets_venn_diagram()