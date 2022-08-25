import seaborn as sns 
import matplotlib.pyplot as plt
from accuracy import get_blimp_acc
from pdb import set_trace as bp
import pandas as pd 
import os

output_root = "output"
plot_dir = "plot_result"
sns.set_theme(style="whitegrid")

model_label2dir = {
    "mp0.15": "roberta.base.faststatsync.me_fp16.cmpltsents.mp0.15.roberta_base.tps512.adam.fp16adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms32.uf4.mu500000.s1.ngpu64",
    "mp0.2": "roberta.base.faststatsync.me_fp16.cmpltsents.mp0.2.roberta_base.tps512.adam.fp16adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms32.uf2.mu500000.s1.ngpu128",
    "mp0.09-0.21": "roberta.base.faststatsync.me_fp16.none.mpr0.09-0.21.roberta_base.tps512.adam.fp16adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms32.uf2.mu500000.s1.ngpu128",
    "mp0.09-0.21_bugfree": "roberta.base.faststatsync.me_fp16.cmpltsents.mpr0.09-0.21.roberta_base.tps512.adam.fp16adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms32.uf2.mu500000.s1.ngpu128",
    "mp0.4": "roberta.base.faststatsync.me_fp16.cmpltsents.mp0.4.roberta_base.tps512.adam.fp16adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms32.uf4.mu500000.s1.ngpu64",
    "mp0.5": "roberta.base.faststatsync.me_fp16.cmpltsents.mp0.5.roberta_base.tps512.adam.fp16adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms32.uf4.mu500000.s1.ngpu64",
    "seq-len[2-8]": "roberta.base.faststatsync.me_fp16.cmpltsents.seq-len.slgeos.slb0.2-0.8.roberta_base.tps512.adam.fp16adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms32.uf4.mu500000.s1.ngpu64",
    "seq-len[3-7]": "roberta.base.faststatsync.me_fp16.cmpltsents.seq-len.slgeos.slb0.3-0.7.roberta_base.tps512.adam.fp16adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms32.uf4.mu500000.s1.ngpu64",
}

if __name__ == "__main__":
    header = ["sample_strategy", "task", "acc"]
    table = []
    for model_label, dirname in model_label2dir.items():
        model_results = get_blimp_acc(f"{output_root}/{dirname}")
        model_results = [[model_label] + r for r in model_results]
        table.extend(model_results)
    df = pd.DataFrame(table, columns=header)
    g = sns.FacetGrid(df, sharey=False, col="task", col_wrap=4)
    
    x = "sample_strategy"
    y = "acc"
    g.map_dataframe(sns.barplot, 
                    x=x, y=y, linewidth=1)
    g.set(ylim=0.6,)
    # g.set_xticklabels(x_ticks, rotation=30)
    g.set_xticklabels(rotation=90)
    g.add_legend()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/all_tasks.x={x}.y={y}.blimp_acc.png")
    plt.clf()
    
    #plot with confidence interval
    task_only_table = df[~df['task'].isin(["overall"])]
    ax = sns.barplot(x=x, y=y, data=task_only_table)
    ax.set(ylim=0.6,)
    plt.savefig(f"{plot_dir}/confidence_interval.x={x}.y={y}.blimp_acc.png")
    