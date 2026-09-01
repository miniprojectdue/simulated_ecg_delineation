"""Step 1/2 extract peak positions for the classes/leads needed by the Figure 6 amplitude
panels (MI, LAE, FAM vs sinus) -> fig6_amp_positions.csv (small)."""
import pandas as pd, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT,"config","paths.yaml")):
    ROOT = os.path.dirname(ROOT)
DATA = os.path.join(HERE, "..", "data")
FIG  = os.path.join(HERE, "..", "figures")
FID  = os.path.join(ROOT,"ecgdeli_labelling","data","primary","medalcare_fiducials_ecgdeli.csv")
CLASSES = {"sinus","mi","lae","fam"}; LEADS = {"II","V2","aVL","V6"}
cols = ["disease_class","record_id","lead","qrs_onset_sample",
        "p_peak_sample","q_peak_sample","r_peak_sample"]
df = pd.read_csv(FID, usecols=cols, na_values=["","None"])
df = df[df.disease_class.isin(CLASSES) & df.lead.isin(LEADS)]
df.to_csv(os.path.join(DATA,"fig6_amp_positions.csv"), index=False)
print("fig6_amp_positions.csv rows:", len(df), "| classes:", sorted(df.disease_class.unique()),
      "| leads:", sorted(df.lead.unique()))
