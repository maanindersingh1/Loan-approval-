import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

R = {}
df = pd.read_csv("data/loan_data.csv")
R["shape"] = df.shape
R["missing"] = df.isna().sum()[lambda s: s > 0].sort_values(ascending=False).to_dict()
R["dups"] = int(df.duplicated().sum())
vc = df.Loan_Status.value_counts(); R["target"] = vc.to_dict()
R["term_360"] = int((df.Loan_Amount_Term == 360).sum()); R["term_known"] = int(df.Loan_Amount_Term.notna().sum())
R["max_income"] = int(df.ApplicantIncome.max())
R["credit_rate"] = (df.groupby("Credit_History").Loan_Status.apply(lambda s: (s=="Y").mean()*100)).round(0).to_dict()
R["area_rate"] = (df.groupby("Property_Area").Loan_Status.apply(lambda s: (s=="Y").mean()*100)).round(0).to_dict()

# EDA figures (before cleaning)
sns.set_theme(style="whitegrid")
BLUE, ORG, GRN = "#2f6fd0", "#ee6c35", "#1fa97a"
fig, ax = plt.subplots(figsize=(5,3.4)); c = df.Loan_Status.value_counts()
ax.bar(["Approved (Y)","Rejected (N)"], [c["Y"], c["N"]], color=[GRN, ORG])
for i,v in enumerate([c["Y"], c["N"]]): ax.text(i, v+5, f"{v} ({v/len(df)*100:.1f}%)", ha="center")
ax.set_ylabel("Applications"); ax.set_title("Loan Status Distribution"); plt.tight_layout(); plt.savefig("outputs/fig1_target.png", dpi=170); plt.close()

d2 = df.dropna(subset=["Credit_History"]).copy()
ct = d2.groupby("Credit_History").Loan_Status.apply(lambda s:(s=="Y").mean()*100)
fig, ax = plt.subplots(figsize=(5,3.4)); ax.bar(["Bad (0)","Good (1)"], ct.values, color=[ORG, GRN])
for i,v in enumerate(ct.values): ax.text(i, v+1.5, f"{v:.0f}%", ha="center")
ax.set_ylim(0,100); ax.set_ylabel("% Approved"); ax.set_title("Approval Rate by Credit History"); plt.tight_layout(); plt.savefig("outputs/fig2_credit.png", dpi=170); plt.close()

pa = df.groupby("Property_Area").Loan_Status.apply(lambda s:(s=="Y").mean()*100).reindex(["Rural","Semiurban","Urban"])
fig, ax = plt.subplots(figsize=(5,3.4)); ax.bar(pa.index, pa.values, color=BLUE)
for i,v in enumerate(pa.values): ax.text(i, v+1.5, f"{v:.0f}%", ha="center")
ax.set_ylim(0,100); ax.set_ylabel("% Approved"); ax.set_title("Approval Rate by Property Area"); plt.tight_layout(); plt.savefig("outputs/fig3_area.png", dpi=170); plt.close()

fig, axs = plt.subplots(1,2, figsize=(8,3.4))
sns.boxplot(y=df.ApplicantIncome, ax=axs[0], color="#9bbcf0"); axs[0].set_title("Applicant Income")
sns.boxplot(y=df.LoanAmount, ax=axs[1], color="#f4b08f"); axs[1].set_title("Loan Amount (thousands)")
plt.tight_layout(); plt.savefig("outputs/fig4_outliers.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(6,3.4))
for s,col,l in [("Y",GRN,"Approved"),("N",ORG,"Rejected")]:
    sns.kdeplot(np.log1p(df[df.Loan_Status==s].ApplicantIncome), ax=ax, color=col, label=l, fill=True, alpha=.3)
ax.set_xlabel("log(Applicant Income)"); ax.set_title("Income Distribution by Loan Status"); ax.legend(); plt.tight_layout(); plt.savefig("outputs/fig5_income.png", dpi=170); plt.close()

# Preprocessing
d = df.drop(columns="Loan_ID").copy()
for c_ in ["Gender","Married","Dependents","Self_Employed","Credit_History"]: d[c_] = d[c_].fillna(d[c_].mode()[0])
d["LoanAmount"] = d.LoanAmount.fillna(d.LoanAmount.median())
d["Loan_Amount_Term"] = d.Loan_Amount_Term.fillna(d.Loan_Amount_Term.mode()[0])
d["Dependents"] = d.Dependents.replace("3+","3").astype(int)
d["Gender"] = (d.Gender=="Male").astype(int); d["Married"] = (d.Married=="Yes").astype(int)
d["Education"] = (d.Education=="Graduate").astype(int); d["Self_Employed"] = (d.Self_Employed=="Yes").astype(int)
y = (d.Loan_Status=="Y").astype(int); d = d.drop(columns="Loan_Status")
d["TotalIncome"] = d.ApplicantIncome + d.CoapplicantIncome
d["TotalIncome_log"] = np.log(d.TotalIncome); d["LoanAmount_log"] = np.log(d.LoanAmount)
d["EMI"] = d.LoanAmount / d.Loan_Amount_Term; d["Loan_to_Income"] = d.LoanAmount / d.TotalIncome
d = pd.concat([d, pd.get_dummies(d.pop("Property_Area"), prefix="Property_Area", drop_first=True).astype(int)], axis=1)

corr_df = d.copy(); corr_df["Loan_Status"] = y
cm_cols = ["Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Loan_Status"]
cm = corr_df[cm_cols].corr()
R["corr_credit"] = round(cm.loc["Credit_History","Loan_Status"],2); R["corr_inc_loan"] = round(cm.loc["ApplicantIncome","LoanAmount"],2)
fig, ax = plt.subplots(figsize=(7,5.2)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size":7}, cbar_kws={"shrink":.8})
ax.set_title("Correlation Heatmap"); plt.tight_layout(); plt.savefig("outputs/fig6_corr.png", dpi=170); plt.close()

feats = ["Gender","Married","Dependents","Education","Self_Employed","Loan_Amount_Term","Credit_History","TotalIncome_log","LoanAmount_log","EMI","Loan_to_Income","Property_Area_Semiurban","Property_Area_Urban"]
X = d[feats]; R["features"] = feats
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
R["baseline"] = round(max(yte.mean(),1-yte.mean())*100,1); R["test_n"] = len(yte)
models = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
          "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
          "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)}
cv = StratifiedKFold(5, shuffle=True, random_state=42); res = {}; cms = {}
for n,m in models.items():
    m.fit(Xtr,ytr); p = m.predict(Xte)
    res[n] = dict(acc=accuracy_score(yte,p)*100, prec=precision_score(yte,p)*100, rec=recall_score(yte,p)*100, f1=f1_score(yte,p), cv=cross_val_score(m,X,y,cv=cv).mean()*100)
    cms[n] = confusion_matrix(yte,p)
R["res"] = res; R["cms"] = {k:v.tolist() for k,v in cms.items()}

mets = ["Accuracy","Precision","Recall","F1-score"]; fig, ax = plt.subplots(figsize=(7,3.6)); w=.25
for i,(n,col) in enumerate(zip(models,[BLUE,ORG,GRN])):
    v = [res[n]["acc"]/100,res[n]["prec"]/100,res[n]["rec"]/100,res[n]["f1"]]; ax.bar(np.arange(4)+(i-1)*w, v, w, label=n, color=col)
ax.axhline(R["baseline"]/100, ls="--", c="k", lw=.8, label="Baseline (approve all)"); ax.set_xticks(range(4)); ax.set_xticklabels(mets); ax.set_ylim(0,1.05)
ax.set_title("Model Comparison (test set)"); ax.legend(ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(.5,-.1), frameon=False); plt.tight_layout(); plt.savefig("outputs/fig7_models.png", dpi=170); plt.close()

fig, axs = plt.subplots(1,3, figsize=(9,3))
for ax,(n,c_) in zip(axs,cms.items()):
    sns.heatmap(c_, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["Rejected","Approved"], yticklabels=["Rejected","Approved"]); ax.set_title(n, fontsize=9); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("outputs/fig8_cm.png", dpi=170); plt.close()

rf = models["Random Forest"]; imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
R["imp"] = imp.sort_values(ascending=False).round(3).to_dict()
fig, ax = plt.subplots(figsize=(6,4)); ax.barh(imp.index, imp.values, color=BLUE); ax.set_title("Feature Importance (Random Forest)"); plt.tight_layout(); plt.savefig("outputs/fig9_imp.png", dpi=170); plt.close()
lr = models["Logistic Regression"].named_steps["logisticregression"]; R["lr_coef"] = dict(zip(feats, lr.coef_[0].round(2)))
json.dump(R, open("results.json","w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
