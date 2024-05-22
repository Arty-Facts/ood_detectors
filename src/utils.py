import numpy as np

def multicol(name, space=2, bars=False):
    if bars:
        return r"\multicolumn{" +str(space)+r"}{|c|}{\textbf{" + name + r"}}"
    return r"\multicolumn{" +str(space)+r"}{c}{\textbf{" + name + r"}}"

def get_names(data, dataset):
    far_names = None
    near_names = None
    for d in data.values():
        if dataset not in d:
            continue
        far_names = []
        near_names = []

        for _dataset in d[dataset]["farood"]:
            if _dataset["dataset"] == "imagenet-o" and dataset != "imagenet_sub":
                continue
            far_names.append(_dataset["dataset"])
       
        for _dataset in d[dataset]["nearood"]:
            if _dataset["dataset"] == "imagenet-o" and dataset != "imagenet_sub":
                continue
            near_names.append(_dataset["dataset"])

        return sorted(far_names), sorted(near_names)
    return far_names, near_names
def create_latex_table(result, method, dataset, encoders):
    far_names, near_names = get_names(result[method], dataset)
    if far_names is None:
        return ""

    total_items = len(far_names) + len(near_names)

    header = r"\begin{table}[ht]"+ "\n"
    header += r"\caption{" +f'Result for '+ method.replace("_", "\\_") + ' on '+ dataset.replace("_", "\\_") + r"}"+ "\n"

    header += r"""\centering
\resizebox{\textwidth}{!}{% Resize table to fit within \textwidth horizontally
"""
    header += r"\begin{tabular}{@{}l*{" + str(total_items+1) + r"}{SS}@{}}" + "\n"
    header += r"\toprule" + "\n"

    description = r"\textbf{Encoder} & "  + " & ".join(multicol(name.replace("_", r"\_")) for name in near_names+far_names ) + " & " + multicol(" Average ") + r" \\" + "\n"
    description +=  r" & {\footnotesize AUROC} $\uparrow$ & {\footnotesize FPR95} $\downarrow$ "*(total_items +1)+ r" \\" + "\n"
    midrule = r"\midrule" + "\n"
    footer = r"\label{tab:" + f"{method}_{dataset}" + r"}" + "\n"
    footer += r"""
\bottomrule
\end{tabular}
}
\end{table}
"""+"\n"

    rows = [" & "   + multicol("Near OOD", len(near_names)*2, True) + " & " + multicol("Far OOD", len(far_names)*2, True) + r" & \\" + "\n"]#" & " + multicol("AVG",2, True) +" & " +multicol("AVG", 2, True) + r" & \\" + "\n"]
    # rows = []

    lookup = {}
    max_data = {}
    for encoder in sorted(encoders):
        if encoder not in result[method]:
            continue
        if dataset not in result[method][encoder]:
            continue
        for name, data in result[method][encoder][dataset].items():
            if name not in ["nearood", "farood"]:
                continue
            lookup_tmp = {d["dataset"]: d["metrics"] for d in data}
            if encoder not in lookup:
                lookup[encoder] = {}
            lookup[encoder].update(lookup_tmp)
 
    for e in encoders:
        if e not in lookup:
            continue
        for data_name in lookup[e]:
            if data_name not in lookup[e]:
                continue
            if data_name not in max_data:
                max_data[data_name] = {
                    "AUC": 0,
                    "FPR_95": 1
                }
            if lookup[e][data_name]["AUC"] > max_data[data_name]["AUC"]:
                max_data[data_name]["AUC"] = lookup[e][data_name]["AUC"]
            if lookup[e][data_name]["FPR_95"] < max_data[data_name]["FPR_95"]:
                max_data[data_name]["FPR_95"] = lookup[e][data_name]["FPR_95"]

    for encoder in sorted(encoders):
        if encoder not in result[method]:
            continue
        if dataset not in result[method][encoder]:
            continue
        if "resnet18_" in encoder:
            row = [r"resnet18\_open\_ood"]
        elif "resnet50_" in encoder:
            row = [r"resnet50\_open\_ood"]
        elif "swin_t" == encoder:
            row = [r"swin\_t\_open\_ood"]
        elif "vit_b16" == encoder:
            row = [r"vit\_b16\_open\_ood"]
        else:
            row = [encoder.replace("_", "\\_")]
        near_auc = 0
        near_fpr = 0
        count = 0
        for data_names in near_names:
            data_res = max_data[data_names]
            max_auc = data_res["AUC"]
            min_fpr = data_res["FPR_95"]
            if data_names in lookup[encoder]:
                metrics = lookup[encoder][data_names]
                if metrics["AUC"] == max_auc:
                    row.append(r"\textbf{" + f"{metrics['AUC']*100:.2f}" + r"}")
                else:
                    row.append(f"{metrics['AUC']*100:.2f}")
                near_auc += metrics["AUC"]
                if metrics["FPR_95"] == min_fpr:
                    row.append(r"\textbf{" + f"{metrics['FPR_95']*100:.2f}" + r"}")
                else:
                    row.append(f"{metrics['FPR_95']*100:.2f}")
                near_fpr += metrics["FPR_95"]
                count += 1
            else:
                row.append("-")
                row.append("-")
        far_auc = 0
        far_fpr = 0
        count2 = 0
        for data_names in far_names:
            data_res = max_data[data_names]
            max_auc = data_res["AUC"]
            min_fpr = data_res["FPR_95"]
            if data_names in lookup[encoder]:
                metrics = lookup[encoder][data_names]
                if metrics["AUC"] == max_auc:
                    row.append(r"\textbf{" + f"{metrics['AUC']*100:.2f}" + r"}")
                else:
                    row.append(f"{metrics['AUC']*100:.2f}")
                far_auc += metrics["AUC"]
                if metrics["FPR_95"] == min_fpr:
                    row.append(r"\textbf{" + f"{metrics['FPR_95']*100:.2f}" + r"}")
                else:
                    row.append(f"{metrics['FPR_95']*100:.2f}")
                far_fpr += metrics["FPR_95"]
                count2 += 1
            else:
                row.append("-")
                row.append("-")
        # row.append(f"{near_auc/count*100:.2f}")
        # row.append(f"{near_fpr/count*100:.2f}")
        # row.append(f"{(far_auc)/count2*100:.2f}")
        # row.append(f"{(far_fpr)/count2*100:.2f}")

        row.append(f"{(far_auc+near_auc)/(count+count2)*100:.2f}")
        row.append(f"{(far_fpr+near_fpr)/(count+count2)*100:.2f}")
        rows.append(" & ".join(row) + r" \\")

    return header + description + midrule + "\n".join(rows) + footer

def create_latex_table_mean(results, method, dataset, encoders):
    methods = ['VESDE', 'VPSDE', 'subVPSDE'] 
    experiments = ['results', 'results_v2', 'results_v3']
    result = results[experiments[0]]
    far_names, near_names = get_names(result[method], dataset)
    
    if far_names is None:
        return ""

    total_items = len(far_names) + len(near_names)

    header = r"\begin{table}[ht]"+ "\n"
    header += r"\caption{" +f'Result for '+ method.replace("_", "\\_") + ' on '+ dataset.replace("_", "\\_") + r"}"+ "\n"

    header += r"""\centering
\resizebox{\textwidth}{!}{% Resize table to fit within \textwidth horizontally
"""
    header += r"\begin{tabular}{@{}l*{" + str(total_items+1) + r"}{SS}@{}}" + "\n"
    header += r"\toprule" + "\n"

    description = r"\textbf{Encoder} & "  + " & ".join(multicol(name.replace("_", r"\_")) for name in near_names+far_names ) + " & " + multicol(" Average ") + r" \\" + "\n"
    description +=  r" & {\footnotesize AUROC} $\uparrow$ & {\footnotesize FPR95} $\downarrow$ "*(total_items +1)+ r" \\" + "\n"
    midrule = r"\midrule" + "\n"
    footer = r"\label{tab:" + f"{method}_{dataset}" + r"}" + "\n"
    footer += r"""
\bottomrule
\end{tabular}
}
\end{table}
"""+"\n"

    rows = [" & "   + multicol("Near OOD", len(near_names)*2, True) + " & " + multicol("Far OOD", len(far_names)*2, True) + r" & \\" + "\n"]#" & " + multicol("AVG",2, True) +" & " +multicol("AVG", 2, True) + r" & \\" + "\n"]
    # rows = []

    lookup = {}
    max_data = {}
    for _experiment in experiments:
        #print(_experiment)
        _result = results[_experiment]
        for _method in methods:
            for encoder in sorted(encoders):
                if _method not in _result:
                    continue
                if encoder not in _result[_method]:
                    continue
                if dataset not in _result[_method][encoder]:
                    continue
                for name, data in _result[_method][encoder][dataset].items():
                    if name not in ["nearood", "farood"]:
                        continue
                    if encoder not in lookup:
                        lookup[encoder] = {}
                    for d in data:
                        if d["dataset"] not in lookup[encoder]:
                            lookup[encoder][d["dataset"]] = {}
                        if "AUC" not in lookup[encoder][d["dataset"]]:
                            lookup[encoder][d["dataset"]]["AUC"] = []
                        lookup[encoder][d["dataset"]]["AUC"].append(d["metrics"]["AUC"])
                        if "FPR_95" not in lookup[encoder][d["dataset"]]:
                            lookup[encoder][d["dataset"]]["FPR_95"] = []
                        lookup[encoder][d["dataset"]]["FPR_95"].append(d["metrics"]["FPR_95"])

    # print(lookup)
    for e in encoders:
        if e not in lookup:
            continue
        for data_name in lookup[e]:
            if data_name not in lookup[e]:
                continue
            if data_name not in max_data:
                max_data[data_name] = {
                    "AUC": 0,
                    "FPR_95": 1
                }
            if np.mean(lookup[e][data_name]["AUC"]) > max_data[data_name]["AUC"]:
                max_data[data_name]["AUC"] = np.mean(lookup[e][data_name]["AUC"])
            if np.mean(lookup[e][data_name]["FPR_95"]) < max_data[data_name]["FPR_95"]:
                max_data[data_name]["FPR_95"] = np.mean(lookup[e][data_name]["FPR_95"])

    for encoder in sorted(encoders):
        if encoder not in result[method]:
            continue
        if dataset not in result[method][encoder]:
            continue
        if "resnet18_" in encoder:
            row = [r"resnet18\_open\_ood"]
        elif "resnet50_" in encoder:
            row = [r"resnet50\_open\_ood"]
        elif "swin_t" == encoder:
            row = [r"swin\_t\_open\_ood"]
        elif "vit_b16" == encoder:
            row = [r"vit\_b16\_open\_ood"]
        else:
            row = [encoder.replace("_", "\\_")]
        near_auc = 0
        near_fpr = 0
        count = 0
        for data_names in near_names+far_names:
            if data_names not in max_data:
                row.append("-")
                row.append("-")
                continue
            data_res = max_data[data_names]
            max_auc = data_res["AUC"]
            min_fpr = data_res["FPR_95"]
            if data_names in lookup[encoder]:
                metrics = lookup[encoder][data_names]
                auc_mean = np.mean(metrics['AUC'])
                auc_std = np.std(metrics['AUC'])
                if auc_mean == max_auc:
                    row.append(r"\textbf{" + f"{auc_mean*100:.2f}±{auc_std*100:.2f}" + r"}")
                else:
                    row.append(f"{auc_mean*100:.2f}±{auc_std*100:.2f}")
                near_auc += auc_mean
                fpr_mean = np.mean(metrics['FPR_95'])
                fpr_std = np.std(metrics['FPR_95'])
                if fpr_mean == min_fpr:
                    row.append(r"\textbf{" + f"{fpr_mean*100:.2f}±{fpr_std*100:.2f}" + r"}")
                else:
                    row.append(f"{fpr_mean*100:.2f}±{fpr_std*100:.2f}")
                near_fpr += fpr_mean
                count += 1
            else:
                row.append("-")
                row.append("-")
       
        # row.append(f"{near_auc/count*100:.2f}")
        # row.append(f"{near_fpr/count*100:.2f}")
        # row.append(f"{(far_auc)/count2*100:.2f}")
        # row.append(f"{(far_fpr)/count2*100:.2f}")
        if count == 0:
            row.append("-")
            row.append("-")
        else:
            row.append(f"{(near_auc)/(count)*100:.2f}")
            row.append(f"{(near_fpr)/(count)*100:.2f}")
        rows.append(" & ".join(row) + r" \\")

    return header + description + midrule + "\n".join(rows) + footer



def create_latex_table2(result, method, dataset, encoders):
    far_names, near_names = get_names(result[method], dataset)
    if far_names is None:
        return ""

    total_items = len(far_names) + len(near_names)

    header = r"\begin{table}[ht]"+ "\n"
    header += r"\caption{" +f'Result for ??? on '+ dataset.replace("_", "\\_") + r"}"+ "\n"

    header += r"""\centering
\resizebox{\textwidth}{!}{% Resize table to fit within \textwidth horizontally
"""
    header += r"\begin{tabular}{@{}l*{" + str(total_items+1) + r"}{SS}@{}}" + "\n"
    header += r"\toprule" + "\n"

    description = r"\textbf{Encoder} & "  + " & ".join(multicol(name.replace("_", r"\_")) for name in near_names+far_names ) + " & " + multicol(" Average ") + r" \\" + "\n"
    description +=  r" & {\footnotesize AUROC} $\uparrow$ & {\footnotesize FPR95} $\downarrow$ "*(total_items +1)+ r" \\" + "\n"
    midrule = r"\midrule" + "\n"
    footer = r"\label{tab:" + f"all" + r"}" + "\n"
    footer += r"""
\bottomrule
\end{tabular}
}
\end{table}
"""+"\n"

    # rows = [" & "   + multicol("Near OOD", len(near_names)*2, True) + " & " + multicol("Far OOD", len(far_names)*2, True) + r" & \\" + "\n"]#" & " + multicol("AVG",2, True) +" & " +multicol("AVG", 2, True) + r" & \\" + "\n"]
    rows = []

    lookup = {}
    max_data = {}
    for encoder in sorted(encoders):
        if encoder not in result[method]:
            continue
        if dataset not in result[method][encoder]:
            continue
        for name, data in result[method][encoder][dataset].items():
            if name not in ["nearood", "farood"]:
                continue
            lookup_tmp = {d["dataset"]: d["metrics"] for d in data}
            if encoder not in lookup:
                lookup[encoder] = {}
            lookup[encoder].update(lookup_tmp)
 
    for e in encoders:
        if e not in lookup:
            continue
        for data_name in lookup[e]:
            if data_name not in lookup[e]:
                continue
            if data_name not in max_data:
                max_data[data_name] = {
                    "AUC": 0,
                    "FPR_95": 1
                }
            if lookup[e][data_name]["AUC"] > max_data[data_name]["AUC"]:
                max_data[data_name]["AUC"] = lookup[e][data_name]["AUC"]
            if lookup[e][data_name]["FPR_95"] < max_data[data_name]["FPR_95"]:
                max_data[data_name]["FPR_95"] = lookup[e][data_name]["FPR_95"]

    for encoder in sorted(encoders):
        if encoder not in result[method]:
            continue
        if dataset not in result[method][encoder]:
            continue
        if "resnet18_" in encoder:
            row = [r"resnet18\_open\_ood"]
        elif "resnet50_" in encoder:
            row = [r"resnet50\_open\_ood"]
        elif "swin_t" == encoder:
            row = [r"swin\_t\_open\_ood"]
        elif "vit_b16" == encoder:
            row = [r"vit\_b16\_open\_ood"]
        row = [method.replace("_", "\\_")]
        near_auc = 0
        near_fpr = 0
        count = 0
        for data_names in near_names:
            data_res = max_data[data_names]
            max_auc = data_res["AUC"]
            min_fpr = data_res["FPR_95"]
            if data_names in lookup[encoder]:
                metrics = lookup[encoder][data_names]
                # if metrics["AUC"] == max_auc:
                #     row.append(r"\textbf{" + f"{metrics['AUC']*100:.2f}" + r"}")
                # else:
                row.append(f"{metrics['AUC']*100:.2f}")
                near_auc += metrics["AUC"]
                # if metrics["FPR_95"] == min_fpr:
                #     row.append(r"\textbf{" + f"{metrics['FPR_95']*100:.2f}" + r"}")
                # else:
                row.append(f"{metrics['FPR_95']*100:.2f}")
                near_fpr += metrics["FPR_95"]
                count += 1
            else:
                row.append("-")
                row.append("-")
        far_auc = 0
        far_fpr = 0
        count2 = 0
        for data_names in far_names:
            data_res = max_data[data_names]
            max_auc = data_res["AUC"]
            min_fpr = data_res["FPR_95"]
            if data_names in lookup[encoder]:
                metrics = lookup[encoder][data_names]
                # if metrics["AUC"] == max_auc:
                #     row.append(r"\textbf{" + f"{metrics['AUC']*100:.2f}" + r"}")
                # else:
                row.append(f"{metrics['AUC']*100:.2f}")
                far_auc += metrics["AUC"]
                # if metrics["FPR_95"] == min_fpr:
                #     row.append(r"\textbf{" + f"{metrics['FPR_95']*100:.2f}" + r"}")
                # else:
                row.append(f"{metrics['FPR_95']*100:.2f}")
                far_fpr += metrics["FPR_95"]
                count2 += 1
            else:
                row.append("-")
                row.append("-")
        # row.append(f"{near_auc/count*100:.2f}")
        # row.append(f"{near_fpr/count*100:.2f}")
        # row.append(f"{(far_auc)/count2*100:.2f}")
        # row.append(f"{(far_fpr)/count2*100:.2f}")

        row.append(f"{(far_auc+near_auc)/(count+count2)*100:.2f}")
        row.append(f"{(far_fpr+near_fpr)/(count+count2)*100:.2f}")
        rows.append(" & ".join(row) + r" \\")

    return header + description + midrule , "\n".join(rows) , footer

def create_latex_table3(result, method, dataset, encoders):
    far_names, near_names = get_names(result[method], dataset)
    if far_names is None:
        return ""

    total_items = len(far_names) + len(near_names)

    header = r"\begin{table}[ht]"+ "\n"
    header += r"\caption{" +f'Result for ??? on '+ dataset.replace("_", "\\_") + r"}"+ "\n"

    header += r"""\centering
\resizebox{\textwidth}{!}{% Resize table to fit within \textwidth horizontally
"""
    header += r"\begin{tabular}{@{}l*{" + str(total_items+1) + r"}{SS}@{}}" + "\n"
    header += r"\toprule" + "\n"

    description = r"\textbf{Encoder} & "  + " & ".join(multicol(name.replace("_", r"\_")) for name in near_names+far_names ) + " & " + multicol(" Average ") + r" \\" + "\n"
    description +=  r" & {\footnotesize AUROC} $\uparrow$ & {\footnotesize FPR95} $\downarrow$ "*(total_items +1)+ r" \\" + "\n"
    midrule = r"\midrule" + "\n"
    footer = r"\label{tab:" + f"all" + r"}" + "\n"
    footer += r"""
\bottomrule
\end{tabular}
}
\end{table}
"""+"\n"

    # rows = [" & "   + multicol("Near OOD", len(near_names)*2, True) + " & " + multicol("Far OOD", len(far_names)*2, True) + r" & \\" + "\n"]#" & " + multicol("AVG",2, True) +" & " +multicol("AVG", 2, True) + r" & \\" + "\n"]
    rows = []

    lookup = {}
    max_data = {}
    for encoder in sorted(encoders):
        if encoder not in result[method]:
            continue
        if dataset not in result[method][encoder]:
            continue
        for name, data in result[method][encoder][dataset].items():
            if name not in ["nearood", "farood"]:
                continue
            lookup_tmp = {d["dataset"]: d["metrics"] for d in data}
            if encoder not in lookup:
                lookup[encoder] = {}
            lookup[encoder].update(lookup_tmp)
 
    for e in encoders:
        if e not in lookup:
            continue
        for data_name in lookup[e]:
            if data_name not in lookup[e]:
                continue
            if data_name not in max_data:
                max_data[data_name] = {
                    "AUC": 0,
                    "FPR_95": 1
                }
            if lookup[e][data_name]["AUC"] > max_data[data_name]["AUC"]:
                max_data[data_name]["AUC"] = lookup[e][data_name]["AUC"]
            if lookup[e][data_name]["FPR_95"] < max_data[data_name]["FPR_95"]:
                max_data[data_name]["FPR_95"] = lookup[e][data_name]["FPR_95"]

    for encoder in sorted(encoders):
        if encoder not in result[method]:
            continue
        if dataset not in result[method][encoder]:
            continue
        if "resnet18_" in encoder:
            row = [r"resnet18\_open\_ood"]
        elif "resnet50_" in encoder:
            row = [r"resnet50\_open\_ood"]
        elif "swin_t" == encoder:
            row = [r"swin\_t\_open\_ood"]
        elif "vit_b16" == encoder:
            row = [r"vit\_b16\_open\_ood"]
        row = [method.replace("_", "\\_")+" "+encoder.replace("_", "\\_")]
        near_auc = 0
        near_fpr = 0
        count = 0
        for data_names in near_names:
            data_res = max_data[data_names]
            max_auc = data_res["AUC"]
            min_fpr = data_res["FPR_95"]
            if data_names in lookup[encoder]:
                metrics = lookup[encoder][data_names]
                # if metrics["AUC"] == max_auc:
                #     row.append(r"\textbf{" + f"{metrics['AUC']*100:.2f}" + r"}")
                # else:
                row.append(f"{metrics['AUC']*100:.2f}")
                near_auc += metrics["AUC"]
                # if metrics["FPR_95"] == min_fpr:
                #     row.append(r"\textbf{" + f"{metrics['FPR_95']*100:.2f}" + r"}")
                # else:
                row.append(f"{metrics['FPR_95']*100:.2f}")
                near_fpr += metrics["FPR_95"]
                count += 1
            else:
                row.append("-")
                row.append("-")
        far_auc = 0
        far_fpr = 0
        count2 = 0
        for data_names in far_names:
            data_res = max_data[data_names]
            max_auc = data_res["AUC"]
            min_fpr = data_res["FPR_95"]
            if data_names in lookup[encoder]:
                metrics = lookup[encoder][data_names]
                # if metrics["AUC"] == max_auc:
                #     row.append(r"\textbf{" + f"{metrics['AUC']*100:.2f}" + r"}")
                # else:
                row.append(f"{metrics['AUC']*100:.2f}")
                far_auc += metrics["AUC"]
                # if metrics["FPR_95"] == min_fpr:
                #     row.append(r"\textbf{" + f"{metrics['FPR_95']*100:.2f}" + r"}")
                # else:
                row.append(f"{metrics['FPR_95']*100:.2f}")
                far_fpr += metrics["FPR_95"]
                count2 += 1
            else:
                row.append("-")
                row.append("-")
        # row.append(f"{near_auc/count*100:.2f}")
        # row.append(f"{near_fpr/count*100:.2f}")
        # row.append(f"{(far_auc)/count2*100:.2f}")
        # row.append(f"{(far_fpr)/count2*100:.2f}")

        row.append(f"{(far_auc+near_auc)/(count+count2)*100:.2f}")
        row.append(f"{(far_fpr+near_fpr)/(count+count2)*100:.2f}")
        rows.append(" & ".join(row) + r" \\")

    return header + description + midrule , "\n".join(rows) , footer