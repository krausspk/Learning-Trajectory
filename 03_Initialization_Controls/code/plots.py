"""Four complete, explicitly exploratory control figures; read-only statistics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DATASETS = ("mnist","fashion_mnist","kmnist")
NAMES = {"mnist":"MNIST","fashion_mnist":"Fashion-MNIST","kmnist":"KMNIST"}
DATASET_COLORS = {"mnist":"#4477AA","fashion_mnist":"#CC6677","kmnist":"#C78C19"}
PRIMARY = "normalized_frobenius|total|optimal|canonical"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "pdf.fonttype":42,"savefig.dpi":230,"axes.titleweight":"semibold"})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(frame,dataset,key):
    rows = frame[(frame.dataset==dataset)&(frame.metric_key==key)].sort_values("checkpoint")
    if len(rows)!=21 or not np.array_equal(rows.checkpoint.to_numpy(),np.arange(21)):
        raise ValueError(f"Incomplete 21-checkpoint control: {dataset}, {key}")
    return rows


def _endpoint(frame,dataset,key):
    return _rows(frame,dataset,key).iloc[-1]


def _save(fig,out,name,artifacts):
    for extension in ("png","pdf"):
        path = out/f"{name}.{extension}"
        fig.savefig(path,bbox_inches="tight",facecolor="white")
        artifacts.append({"path":str(path.resolve()),"sha256":sha256(path)})
    plt.close(fig)


def _label(key):
    metric,layer,alignment,gauge = key.split("|")
    layer = "total" if layer=="total" else layer
    metric_name = {"frobenius":"Frobenius","normalized_frobenius":"Unit-norm Frobenius","norm_only":"Norm difference",
                   "small_residual_rms":"Small residual RMS","large_residual_rms":"Large residual RMS",
                   "activity_low_normalized_frobenius":"Low activity","activity_high_normalized_frobenius":"High activity",
                   "norm_rank_difference":"Norm rank difference"}[metric]
    match = {"identity":"native","optimal":"optimal","none":"no match","masked_optimal":"masked optimal"}[alignment]
    label = f"{gauge.capitalize()} | {match} | {metric_name} | {layer}"
    return "Post hoc | "+label if metric=="norm_rank_difference" else label


def _score_limits(frame):
    return min(-.03,float(frame.rank_score_ci_low.min())-.035),max(1.035,float(frame.rank_score_ci_high.max())+.035)


def endpoint_summary(frame,definitions,out,artifacts):
    keys = list(definitions)
    if len(keys) not in (42,48): raise ValueError("The endpoint overview must contain the 42 planned protocols and, when available, all six post hoc norm-rank controls")
    endpoint = frame[frame.checkpoint==20]
    limits = _score_limits(endpoint)
    fig,axes = plt.subplots(1,3,figsize=(16.5,16.0 if len(keys)==42 else 18.0),sharey=True)
    fig.subplots_adjust(left=.36,right=.985,bottom=.075,top=.915,wspace=.16)
    y = np.arange(len(keys))
    reference = keys.index(PRIMARY)
    # Boundary lines separate the prespecified protocol families.
    groups = [(key.split("|")[0],key.split("|")[2],key.split("|")[3]) for key in keys]
    boundaries = [i-.5 for i in range(1,len(keys)) if groups[i]!=groups[i-1]]
    for ax,dataset in zip(axes,DATASETS):
        ordered = pd.DataFrame([_endpoint(frame,dataset,key) for key in keys])
        means = ordered.rank_score.to_numpy(float)
        lo,hi = ordered.rank_score_ci_low.to_numpy(float),ordered.rank_score_ci_high.to_numpy(float)
        if len(keys)==48:ax.axhspan(41.5,47.5,color="#eeeeee",zorder=0)
        ax.axhspan(reference-.48,reference+.48,color="#fff0ba",zorder=0)
        for boundary in boundaries: ax.axhline(boundary,color="#e8e8e8",lw=.7,zorder=0)
        ax.errorbar(means,y,xerr=np.vstack((means-lo,hi-means)),fmt="o",ms=3.8,capsize=2,
                    color=DATASET_COLORS[dataset],ecolor=DATASET_COLORS[dataset],elinewidth=1,zorder=3)
        ax.axvline(.5,color="#777777",ls="--",lw=1.2,zorder=1)
        ax.axvline(1,color="#cccccc",ls=":",lw=.8,zorder=0)
        ax.set(xlim=limits,ylim=(len(keys)-.45,-.65),title=NAMES[dataset],xlabel="Rank score (higher is better)")
        ax.set_xticks([0,.25,.5,.75,1])
        ax.grid(axis="x",alpha=.12)
        ax.tick_params(axis="y",length=0,labelsize=8.5)
    axes[0].set_yticks(y,[_label(key) for key in keys])
    axes[0].get_yticklabels()[reference].set_fontweight("bold")
    axes[0].get_yticklabels()[reference].set_color("#8e6200")
    title = "42 planned controls + 6 post hoc norm-rank controls at E20" if len(keys)==48 else "All 42 initialization-retrieval controls at E20"
    fig.suptitle(title,fontsize=17,y=.978)
    fig.text(.5,.951,"All exploratory | 10 complete trajectories per dataset | yellow = Point 1 reference; gray = post hoc controls",ha="center",fontsize=10.5)
    fig.text(.5,.018,
             "Rank score = (10 - mean true-candidate rank) / 9; chance = 0.5, perfect = 1. Error bars: full-scale 95% delete-one-trajectory jackknife intervals.\n"
             "Intervals are shown without clipping. Zero-width intervals at perfect retrieval do not establish population certainty.\n"
             "Optimal matches use one shared W0/W1 assignment. Gray rows calibrate norm ranks within the full E0/Et ensembles: post hoc, transductive, exploratory.",
             ha="center",fontsize=9)
    _save(fig,out,"control_endpoint_summary",artifacts)


def trajectories(frame,out,artifacts):
    panels = [
        [("normalized_frobenius|total|identity|raw","Raw / native","#3366aa","-"),
         ("normalized_frobenius|total|optimal|raw","Raw / optimal","#bb5577","--"),
         ("normalized_frobenius|total|identity|canonical","Canonical / native","#448866","-."),
         (PRIMARY,"Canonical / optimal (reference)","#222222",":")],
        [("norm_only|total|none|raw","Raw total norm only","#8866aa","-"),
         ("norm_only|total|none|canonical","Canonical total norm only","#bb8833","--"),
         ("activity_low_normalized_frobenius|total|masked_optimal|canonical","Low query activity","#339999","-."),
         ("activity_high_normalized_frobenius|total|masked_optimal|canonical","High query activity","#cc6633",":")],
    ]
    fig,axes = plt.subplots(2,3,figsize=(15.5,9.6),sharex=True,sharey=True)
    fig.subplots_adjust(left=.07,right=.985,bottom=.24,top=.87,hspace=.37,wspace=.19)
    for row,protocols in enumerate(panels):
        for ax,dataset in zip(axes[row],DATASETS):
            trajectories_here = []
            for key,label,color,style in protocols:
                data = _rows(frame,dataset,key)
                x,y = data.checkpoint.to_numpy(),data.rank_score.to_numpy(float)
                trajectories_here.append(y)
                ax.plot(x,y,ls=style,color=color,lw=2 if key==PRIMARY else 1.65,marker="o",ms=2.8,label=label)
                ax.fill_between(x,np.maximum(0,data.rank_score_ci_low.to_numpy(float)),np.minimum(1,data.rank_score_ci_high.to_numpy(float)),color=color,alpha=.065)
            ax.axhline(.5,color="#777777",ls="--",lw=1)
            ax.set(title=f"{NAMES[dataset]} | {'coordinate and gauge' if row==0 else 'norm and activity'}",ylim=(-.03,1.06),xlim=(-.35,20.35))
            ax.set_xticks([0,1,2,5,10,15,20]);ax.grid(alpha=.15)
            if any(np.allclose(trajectories_here[a],trajectories_here[b],atol=1e-12,rtol=0) for a in range(4) for b in range(a+1,4)):
                ax.text(.98,.07,"Some protocol curves coincide",ha="right",transform=ax.transAxes,fontsize=8,color="#555555")
            if row==1:ax.set_xlabel("Checkpoint E (half epochs)")
    for ax in axes[:,0]:ax.set_ylabel("Rank score (chance = 0.5)")
    handles = [Line2D([0],[0],color=color,ls=style,lw=2,label=label) for protocols in panels for _,label,color,style in protocols]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.51,.105),ncol=4,frameon=False,fontsize=8.7,columnspacing=1.6)
    fig.suptitle("Prespecified total-network control trajectories",fontsize=16,y=.974)
    fig.text(.5,.934,"Four coordinate/gauge protocols, two norm-only controls, and low/high query-activity controls; all exploratory",ha="center",fontsize=10)
    fig.text(.5,.026,
             "Bands show 95% trajectory-level jackknife intervals, clipped to [0,1] for this display; CSVs retain full intervals. E0 is sanity only.\n"
             "Zero-width intervals at perfect retrieval are not population certainty. These eight protocols were prespecified; all 42 planned controls appear in the endpoint overview.\n"
             "Absolute norm difference can induce identical candidate ordering; it does not exclude all norm-based lineage information. Additional norm-rank controls are explicitly post hoc.",
             ha="center",fontsize=8.8)
    _save(fig,out,"control_trajectories",artifacts)


def _layer_endpoint_panel(ax,frame,dataset,protocols,limits):
    positions = np.arange(3)
    offsets = np.linspace(-.18,.18,len(protocols))
    for offset,(template,label,color,marker) in zip(offsets,protocols):
        data = [_endpoint(frame,dataset,template.format(layer=layer)) for layer in ("W0","W1","total")]
        values = np.asarray([row.rank_score for row in data])
        lo = np.asarray([row.rank_score_ci_low for row in data])
        hi = np.asarray([row.rank_score_ci_high for row in data])
        ax.errorbar(positions+offset,values,yerr=np.vstack((values-lo,hi-values)),fmt=marker,ms=6,capsize=3,
                    color=color,elinewidth=1.1,label=label)
    ax.axhline(.5,color="#777777",ls="--",lw=1)
    ax.set_xticks(positions,["W0","W1","Total"])
    ax.set(ylim=limits,xlim=(-.5,2.5),title=NAMES[dataset],xlabel="Weight contribution")
    ax.grid(axis="y",alpha=.15)


def layer_activity(frame,out,artifacts):
    counts = pd.read_csv(out/"data/query_mask_counts.csv")
    inactive = counts[(counts.checkpoint==20)&(counts.subset=="inactive")]
    protocols = [("normalized_frobenius|{layer}|optimal|canonical","All neurons / reference","#333333","o"),
                 ("activity_low_normalized_frobenius|{layer}|masked_optimal|canonical","Low query activity","#339999","s"),
                 ("activity_high_normalized_frobenius|{layer}|masked_optimal|canonical","High query activity","#cc6633","^")]
    selected_keys = [template.format(layer=layer) for template,*_ in protocols for layer in ("W0","W1","total")]
    selected = frame[(frame.checkpoint==20)&frame.metric_key.isin(selected_keys)]
    limits = _score_limits(selected)
    fig,axes = plt.subplots(2,3,figsize=(15.1,9.1))
    fig.subplots_adjust(left=.075,right=.985,bottom=.18,top=.82,hspace=.63,wspace=.27)
    maximum = max(1,float(inactive.selected_neurons.max())+1)
    for ax,dataset in zip(axes[0],DATASETS):
        _layer_endpoint_panel(ax,frame,dataset,protocols,limits)
        texts = []
        for subset in ("low","high"):
            values = counts[(counts.dataset==dataset)&(counts.checkpoint==20)&(counts.subset==subset)].selected_neurons
            texts.append(f"{subset}: {values.min()}-{values.max()}")
        ax.text(.5,-.28,"Selected neurons per query: "+"; ".join(texts),transform=ax.transAxes,ha="center",fontsize=8.5)
    for ax,dataset in zip(axes[1],DATASETS):
        table = inactive[inactive.dataset==dataset].copy()
        table["order"] = table.run_id.str[1:].astype(int)
        table = table.sort_values("order")
        x = np.arange(10)
        ax.bar(x,table.selected_neurons,color=DATASET_COLORS[dataset],alpha=.78,width=.7)
        for i,value in enumerate(table.selected_neurons):ax.text(i,value+.03*maximum,str(int(value)),ha="center",fontsize=8)
        ax.set_xticks(x,table.run_id,rotation=45)
        ax.set(ylim=(0,maximum),title=f"{NAMES[dataset]} | probe-inactive neurons",xlabel="Individual E20 query network")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True));ax.grid(axis="y",alpha=.15)
        if table.selected_neurons.eq(0).all():
            ax.text(.5,.56,"0 / 64 probe-inactive neurons\nin every E20 query",transform=ax.transAxes,ha="center",va="center",fontsize=11,color="#555555")
    axes[0,0].set_ylabel("Rank score (chance = 0.5)")
    axes[1,0].set_ylabel("Number of inactive hidden neurons / 64")
    handles = [Line2D([0],[0],color=color,marker=marker,lw=0,label=label,ms=6) for _,label,color,marker in protocols]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.9),ncol=3,frameon=False,fontsize=10)
    fig.suptitle("Layer contributions and query-activity controls at E20",fontsize=16,y=.977)
    fig.text(.5,.935,"Exploratory retrieval controls above; all 30 individual probe-inactive neuron counts below",ha="center",fontsize=10)
    fig.text(.5,.023,
             "All weights share one canonical, globally unit-normalized vector. Query-only occupancy quartiles retain boundary ties; masked Hungarian is recomputed for every candidate.\n"
             "Inactive means zero activation on the fixed 3,000-image probe, not globally dead. Error bars: 95% trajectory-level jackknife; perfect-score degeneracy is not population certainty.\n"
             "Layer retrieval depends on alignment: weak W1 retrieval under a joint optimal assignment does not imply that all W1 lineage information was erased.",
             ha="center",fontsize=8.7)
    _save(fig,out,"layer_activity_controls",artifacts)


def residual_and_change(frame,out,artifacts):
    history = pd.read_csv(out/"data/historical_weight_changes_descriptive.csv")
    if history.used_for_retrieval.astype(str).str.lower().eq("true").any():
        raise ValueError("Historical true-parent changes must not enter retrieval controls")
    protocols = [("normalized_frobenius|{layer}|optimal|canonical","Full normalized distance","#333333","o"),
                 ("small_residual_rms|{layer}|optimal|canonical","Small candidate-pair residuals","#5577bb","s"),
                 ("large_residual_rms|{layer}|optimal|canonical","Large candidate-pair residuals","#bb6688","^")]
    selected_keys = [template.format(layer=layer) for template,*_ in protocols for layer in ("W0","W1","total")]
    limits = _score_limits(frame[(frame.checkpoint==20)&frame.metric_key.isin(selected_keys)])
    fig,axes = plt.subplots(2,3,figsize=(15.3,10.2))
    fig.subplots_adjust(left=.075,right=.985,bottom=.18,top=.81,hspace=.73,wspace=.27)
    for ax,dataset in zip(axes[0],DATASETS):_layer_endpoint_panel(ax,frame,dataset,protocols,limits)
    percentiles = [25,50,75,90,95,99]
    for ax,dataset in zip(axes[1],DATASETS):
        for layer,color in (("W0","#3366aa"),("W1","#cc8844")):
            subset = history[(history.dataset==dataset)&(history.checkpoint==20)&(history.layer==layer)]
            values = subset[[f"abs_change_p{p}" for p in percentiles]].to_numpy(float)
            if len(values)!=10:raise ValueError("Historical summary must retain ten complete runs")
            ax.plot(percentiles,np.median(values,axis=0),"o-",ms=3.5,lw=1.6,color=color,label=layer)
            ax.fill_between(percentiles,values.min(axis=0),values.max(axis=0),color=color,alpha=.13)
        ax.set(title=NAMES[dataset],xlabel="Within-run absolute-change percentile",xlim=(23,101),ylim=(0,None))
        ax.set_xticks([25,50,75,90,99]);ax.grid(alpha=.15)
        ax.legend(loc="upper left",frameon=False,fontsize=9)
    axes[0,0].set_ylabel("Rank score (chance = 0.5)")
    axes[1,0].set_ylabel("Raw |W(E20) - W(E0)|")
    handles = [Line2D([0],[0],color=color,marker=marker,lw=0,label=label,ms=6) for _,label,color,marker in protocols]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.89),ncol=3,frameon=False,fontsize=9.5)
    fig.suptitle("Residual sensitivity and historical change are different analyses",fontsize=16,y=.978)
    fig.text(.5,.936,"A  E20 retrieval using candidate-specific residual quartiles (exploratory)",ha="center",fontsize=11,weight="semibold")
    fig.text(.5,.482,"B  True-parent raw weight changes (descriptive only; never used for retrieval)",ha="center",fontsize=11,weight="semibold")
    fig.text(.5,.022,
             "A: bottom/top quartiles are recomputed separately for every query/candidate pair after shared matching, per layer, including all boundary ties; total pools selected entries.\n"
             "This is metric sensitivity, not a causal ablation. Error bars: 95% trajectory-level jackknife; perfect-score degeneracy is not population certainty.\n"
             "B: lines are medians of the ten run-specific change quantiles; shading is the full run range, not a confidence interval. Native raw coordinates, no rematching.",
             ha="center",fontsize=8.6)
    _save(fig,out,"residual_and_change_controls",artifacts)


def make_plots(out:Path):
    out = Path(out).resolve()
    if out.name!="03_Initialization_Controls":raise ValueError("These plots belong only to 03_Initialization_Controls")
    table = pd.read_csv(out/"retrieval_by_dataset.csv")
    definitions_map = json.loads((out/"data/distance_definitions.json").read_text())
    if set(table.metric_key)!=set(definitions_map):raise ValueError("Statistics and the 42 planned definitions differ")
    endpoint_table,endpoint_definitions = table,dict(definitions_map)
    posthoc_path = out/"norm_rank_controls/retrieval_by_dataset.csv"
    extra_sources = []
    if posthoc_path.exists():
        posthoc = pd.read_csv(posthoc_path)
        if posthoc.metric_key.nunique()!=6 or not posthoc.metric.eq("norm_rank_difference").all():
            raise ValueError("Expected exactly the six declared post hoc norm-rank protocols")
        for key in posthoc.metric_key.drop_duplicates():endpoint_definitions[key] = {"post_hoc":True}
        endpoint_table = pd.concat([table,posthoc],ignore_index=True)
        extra_sources = [posthoc_path,out/"norm_rank_controls/analysis_note.json"]
    artifacts = []
    endpoint_summary(endpoint_table,endpoint_definitions,out,artifacts)
    trajectories(table,out,artifacts)
    layer_activity(table,out,artifacts)
    residual_and_change(table,out,artifacts)
    sources = [out/"retrieval_by_dataset.csv",out/"data/distance_definitions.json",out/"data/query_mask_counts.csv",
               out/"data/historical_weight_changes_descriptive.csv",Path(__file__)]+extra_sources
    receipt = {"plot_code_sha256":sha256(__file__),"sources":{str(p):sha256(p) for p in sources},"artifacts":artifacts,
               "all_42_controls_in_endpoint_overview":True,"n_post_hoc_controls_in_overview":len(endpoint_definitions)-42,"all_controls_exploratory":True,
               "selected_trajectory_protocols_fixed_in_code":True,"historical_changes_not_used_for_retrieval":True,
               "visual_review_pending":True}
    path = out/"provenance/plot_sources.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    receipt = make_plots(args.output)
    print(json.dumps({"artifacts":len(receipt["artifacts"]),"source_files":len(receipt["sources"])}))
