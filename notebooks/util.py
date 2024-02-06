import torch
import matplotlib.pyplot as plt


def calculate_intervention_posteriors(model, loader, device):
    intervention_posteriors = []
    intervention_labels = []

    for i, (x1, x2, _, _, intervention_label, *_) in enumerate(loader):
        print(f"Batch {i + 1} / {len(loader)}", end="\r")

        x1 = x1.to(device)
        x2 = x2.to(device)
        intervention_label = intervention_label.to(device)

        e1_mean, e1_std = model.encoder.mean_std(x1)
        e2_mean, e2_std = model.encoder.mean_std(x2)
        intervention_encoder_inputs = torch.cat((e1_mean, e2_mean - e1_mean), dim=1)
        intervention_posteriors.append(model.intervention_encoder(intervention_encoder_inputs))
        intervention_labels.append(intervention_label)
    intervention_posteriors = torch.cat(intervention_posteriors, dim=0)
    intervention_labels = torch.cat(intervention_labels, dim=0)
    print()
    return intervention_posteriors, intervention_labels


def calculate_intervention_posterior_heatmap(
    intervention_posteriors, intervention_labels, dim_z, device
):
    num_interventions = dim_z + 1

    heatmap = torch.zeros(num_interventions, num_interventions)
    heatmap = heatmap.to(device)

    for intervention_posterior, intervention_label in zip(
        intervention_posteriors,
        intervention_labels.int(),  # int() because indexing with uint gives wrong results
    ):
        heatmap[intervention_label] += intervention_posterior
    heatmap /= heatmap.sum(dim=1, keepdim=True)
    return heatmap


def plot_intervention_posterior_heatmap(heatmap, dim_z, ax=None):
    num_interventions = dim_z + 1
    if ax is None:
        fig, ax = plt.subplots()

    img = ax.imshow(heatmap.cpu().detach().numpy())
    ax.set_xticks(range(num_interventions))
    ax.set_yticks(range(num_interventions))
    ax.set_xticklabels(["empty"] + [f"$\widehat{{z_{i + 1}}}$" for i in range(dim_z)])
    ax.set_yticklabels(["empty"] + [f"$z_{i + 1}$" for i in range(dim_z)])
    ax.set_xlabel("Predicted Intervention on")
    ax.set_ylabel("True Intervention on")
    ax.set_title("Intervention Encoder Posterior Heatmap")
    cbar = ax.figure.colorbar(img, ax=ax)
    cbar.ax.set_ylabel("Posterior", rotation=-90, va="bottom")
