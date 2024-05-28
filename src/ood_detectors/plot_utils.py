import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pathlib


def plot(eval_data, id_name, ood_names, encoder, model, out_dir='figs', config=None, verbose=True, train_loss=None, ext="png"):
    if verbose:
        print('Generating plots...')
    # Unpack eval_data
    score, score_ref = eval_data['score'], eval_data['score_ref']
    ref_auc, ref_fpr = eval_data['ref_auc'], eval_data['ref_fpr']
    score_oods, auc_oods, fpr_oods = eval_data['score_oods'], eval_data['auc'], eval_data['fpr']

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))  # Adjust the size as needed
    fig.suptitle(f'{model} Evaluation on {encoder}')

    def add_shadow(ax, data): 
        if data.var() > 1e-6:
            l = ax.lines[-1]
            x = l.get_xydata()[:,0]
            y = l.get_xydata()[:,1]
            ax.fill_between(x,y, alpha=0.1)
            # Calculate and plot the mean
            mean_value = np.mean(data)
            line_color = l.get_color()
            ax.axvline(mean_value, color=line_color, linestyle=':', linewidth=1.5)
    # Subplot 1: KDE plots
    if len(score.shape) > 1:
        sns.kdeplot(data=np.mean(score, axis=0), bw_adjust=.2, ax=axs[0, 0], label=f'{id_name} training: {np.mean(np.mean(score, axis=0)):.2f}')
        add_shadow(axs[0, 0], np.mean(score, axis=0))
    else:
        sns.kdeplot(data=score, bw_adjust=.2, ax=axs[0, 0], label=f'{id_name} training: {np.mean(score):.2f}')
        add_shadow(axs[0, 0], score)

    if len(score_ref.shape) > 1:
        sns.kdeplot(data=np.mean(score_ref, axis=0), bw_adjust=.2, ax=axs[0, 0], label=f'{id_name} validation: {np.mean(np.mean(score_ref, axis=0)):.2f}')
        add_shadow(axs[0, 0], np.mean(score_ref, axis=0))
    else:
        sns.kdeplot(data=score_ref, bw_adjust=.2, ax=axs[0, 0], label=f'{id_name} validation: {np.mean(score_ref):.2f}')
        add_shadow(axs[0, 0], score_ref)

    for ood_name, score_ood in zip(ood_names, score_oods):
        if len(score_ood.shape) > 1:
            sns.kdeplot(data=np.mean(score_ood, axis=0), bw_adjust=.2, ax=axs[0, 0], label=f'{ood_name}: {np.mean(np.mean(score_ood, axis=0)):.2f}')
            add_shadow(axs[0, 0], np.mean(score_ood, axis=0))
        else:
            sns.kdeplot(data=score_ood, bw_adjust=.2, ax=axs[0, 0], label=f'{ood_name}: {np.mean(score_ood):.2f}')
            add_shadow(axs[0, 0], score_ood)
    # axs[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
    # axs[0, 0].xaxis.set_minor_locator(ticker.MultipleLocator(base=0.1))
    axs[0, 0].set_title('Density Plots')
    axs[0, 0].set_xlabel('bits/dim')
    axs[0, 0].set_ylabel('Density')
    # axs[0, 0].set_xlim(6.5, 8)
    # axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Subplot 2: Bar chart for AUC and FPR
    x = np.arange(len(ood_names)+1)  # the label locations
    width = 0.35  # the width of the bars
    disp_auc = [ref_auc] + auc_oods
    disp_fpr = [ref_fpr] + fpr_oods
    rects1 = axs[0, 1].bar(x - width/2, disp_auc, width, label='AUC', alpha=0.6)
    rects2 = axs[0, 1].bar(x + width/2, disp_fpr, width, label='FPR', alpha=0.6)
    axs[0, 1].set_ylabel('Metric Value')
    axs[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(base=0.1))
    axs[0, 1].yaxis.set_minor_locator(ticker.MultipleLocator(base=0.05))
    axs[0, 1].set_title(f'AUC and FPR Metrics\nMean AUC: {np.mean(disp_auc[1:]):.2f}, Mean FPR: {np.mean(disp_fpr[1:]):.2f}')
    axs[0, 1].set_xticks(x)
    names = [f'{name}\nAUC: {auc:.2f}\nFPR: {fpr:.2f}' for name, auc, fpr in zip([id_name]+list(ood_names), disp_auc, disp_fpr)]
    axs[0, 1].set_xticklabels(names)
    axs[0, 1].grid(True)
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].legend()
    # add line at 0.5
    axs[0, 1].axhline(0.5, color='red', linestyle='--', linewidth=1.5) 

    if train_loss is not None:
        # Subplot 3: Training loss over time
        axs[1, 0].plot(train_loss, label='Training Loss')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].set_title('Training Loss Over Time')
        axs[1, 0].grid(True)
        axs[1, 0].legend()
    else:
        axs[1, 0].axis('off')

    # # Subplot 4: Configuration display
    # if config is not None:
    #     config_text = "\n".join([f"{key}: {value}" for key, value in config.items()])
    #     axs[1, 1].text(0.5, 0.5, config_text, ha='center', va='center', fontsize=12, transform=axs[1, 1].transAxes)
    #     axs[1, 1].set_title('Configuration')
    # axs[1, 1].axis('off')
     # Subplot 4: scatter plot of scores
    if score.ndim == 2:
        items, features = score.shape
        score1, score2 = np.mean(score[:items//2], axis=0), np.mean(score[items//2:], axis=0)
        axs[1, 1].scatter(score1, score2, alpha=0.5, label='ID Training', s=1)
        
        score_ref1, score_ref2 = np.mean(score_ref[:items//2], axis=0), np.mean(score_ref[items//2:], axis=0)
        axs[1, 1].scatter(score_ref1, score_ref2, alpha=0.5, label='ID Validation', s=1)

        for ood_name, score_ood in zip(ood_names, score_oods):
            score_ood1, score_ood2 = np.mean(score_ood[:items//2], axis=0), np.mean(score_ood[items//2:], axis=0)
            axs[1, 1].scatter(score_ood1, score_ood2, alpha=0.5, label=ood_name, s=1)
        axs[1, 1].set_xlabel(f'mean bits/dim for first {items//2} models')
        axs[1, 1].set_ylabel(f'mean bits/dim for last {items//2} models')
        axs[1, 1].set_title('Scatter Plot of Scores')
        axs[1, 1].legend()
        
    else:
        axs[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout

    # Save the figure
    out_dir = pathlib.Path(out_dir) / encoder / id_name
    out_dir.mkdir(exist_ok=True, parents=True)
    filename = f"{encoder}_{model}_{id_name}_{int(np.mean(disp_auc[1:])*100)}.{ext}"
    (out_dir / filename).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_dir / filename, bbox_inches='tight')
    if verbose:
        plt.show()