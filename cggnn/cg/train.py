from multiprocessing.sharedctypes import Value
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import seaborn as sb

torch.set_printoptions(sci_mode=False)

def _save_networks(cg_model, u_model, epoch,
                   root_folder_name):

    chkpt_folder_name = root_folder_name + str(epoch) + "/"
    try:
        os.mkdir(chkpt_folder_name)
    except:
        pass
    cg_model.save_networks(chkpt_folder_name)
    u_model.save_networks(chkpt_folder_name)


def _save_proj_matrices(cg_model, val_dataloader, epoch,
                        root_folder_name, save_tag):

    chkpt_folder_name = root_folder_name + str(epoch) + "/"
    try:
        os.mkdir(chkpt_folder_name)
    except:
        pass
    chkpt_folder_name += str(save_tag) + "_"
    all_val_proj = []
    all_val_inv_proj = []
    for (val_x, _, _, batch_size) in val_dataloader:
        proj, inv_proj, = cg_model.get_proj(val_x, batch_size = batch_size)
        all_val_proj.append(proj.detach())
        all_val_inv_proj.append(inv_proj.detach())

    all_val_proj = torch.stack(all_val_proj).view((-1,
                                                   cg_model.cg.num_beads,
                                                   cg_model.cg.num_atoms))
    all_val_inv_proj = torch.stack(all_val_inv_proj).view((-1,
                                                           cg_model.cg.num_atoms_r,
                                                           cg_model.cg.num_beads))

    try:
        mean_proj = all_val_proj.mean(0).cpu().numpy()
        var_proj = all_val_proj.var(0).cpu().numpy()

        num_beads = mean_proj.shape[0]
        beadlabels = [str(i) for i in range(num_beads)]

        sb.heatmap(mean_proj, square=True, xticklabels=cg_model.atom_names,
                   yticklabels=beadlabels,
                   cmap="coolwarm")
        plt.title("Mean Projection (Validation Set)")
        plt.xlabel("Atom")
        plt.ylabel("Bead")
        plt.xticks(fontsize=1)
        plt.yticks(fontsize=1)
        plt.savefig(chkpt_folder_name + "mean_projection.pdf")
        plt.close()

        sb.heatmap(var_proj, square=True, xticklabels=cg_model.atom_names,
                   yticklabels=beadlabels,
                   cmap="coolwarm")
        plt.title("Variance Projection (Validation Set)")
        plt.xlabel("Atom")
        plt.ylabel("Bead")
        plt.xticks(fontsize=1)
        plt.yticks(fontsize=1)
        plt.savefig(chkpt_folder_name + "var_projection.pdf")
        plt.close()

        mean_inv_proj = all_val_inv_proj.mean(0).cpu().numpy()
        var_inv_proj = all_val_inv_proj.var(0).cpu().numpy()

        sb.heatmap(mean_inv_proj, square=True, cmap="coolwarm")
        plt.title("Mean Inverse Projection (Validation Set)")
        plt.xlabel("Atom")
        plt.ylabel("Bead")
        plt.savefig(chkpt_folder_name + "mean_inv_projection.pdf")
        plt.close()

        sb.heatmap(var_inv_proj, square=True, cmap="coolwarm")
        plt.title("Variance Inverse Projection (Validation Set)")
        plt.xlabel("Atom")
        plt.ylabel("Bead")
        plt.savefig(chkpt_folder_name + "var_inv_projection.pdf")
        plt.close()
    except ValueError:
        pass


def _train_cg(cg_model, u_model,
              train_dataloader, val_dataloader, writer, cg_epoch_num, reg,
              add_mean_force_reg=False, detach_link_loss=10000, fix_inv_proj_epoch=0):
    """Carries out cg training
    """
    cg_reg_aux, cg_reg_force = reg
    num_train_batch = len(train_dataloader)
    num_val_batch = len(val_dataloader)
    epoch_train_mse_force_loss = 0
    epoch_val_mse_force_loss = 0
    epoch_train_r_loss = 0
    epoch_val_r_loss = 0
    epoch_train_link_loss = 0
    epoch_val_link_loss = 0
    epoch_train_ent_loss = 0
    epoch_val_ent_loss = 0
    epoch_train_diag_loss = 0
    epoch_val_diag_loss = 0
    for (train_x, train_forces, _, batch_size) in train_dataloader:
        r_loss, mse_force_loss, link_loss, ent_loss, diag_loss = cg_model.get_loss(train_x,
                                                                                   train_forces,
                                                                                   u_model,
                                                                                   batch_size,
                                                                                   add_mean_force_reg=add_mean_force_reg,
                                                                                   fix_inv_proj=(cg_epoch_num<fix_inv_proj_epoch))
        if cg_epoch_num < 5:
            fix_inv_proj = True
        if cg_epoch_num > detach_link_loss:
            link_loss = link_loss.detach()
        loss = (r_loss + cg_reg_force * cg_reg_aux * mse_force_loss +
                cg_reg_aux * link_loss + cg_reg_aux * ent_loss + cg_reg_aux * diag_loss)

        cg_model.cg_optimizer.zero_grad()
        loss.backward()
        cg_model.cg_optimizer.step()

        epoch_train_mse_force_loss += mse_force_loss.item()
        epoch_train_r_loss += r_loss.item()
        epoch_train_link_loss += link_loss.item()
        epoch_train_ent_loss += ent_loss.item()
        epoch_train_diag_loss += diag_loss.item()
    for (val_x, val_forces, _, batch_size) in val_dataloader:
        r_loss, mse_force_loss, link_loss, ent_loss, diag_loss = cg_model.get_loss(val_x,
                                                                                   val_forces,
                                                                                   u_model,
                                                                                   batch_size,
                                                                                   add_mean_force_reg=add_mean_force_reg)
        epoch_val_mse_force_loss += mse_force_loss.item()
        epoch_val_r_loss += r_loss.item()
        epoch_val_link_loss += link_loss.item()
        epoch_val_ent_loss += ent_loss.item()
        epoch_val_diag_loss += diag_loss.item()

    writer.add_scalar("train_cg/mse_force_loss",
                      epoch_train_mse_force_loss / num_train_batch, cg_epoch_num)
    writer.add_scalar("val_cg/mse_force_loss",
                      epoch_val_mse_force_loss / num_val_batch, cg_epoch_num)

    writer.add_scalar("train_cg/r",
                      epoch_train_r_loss / num_train_batch, cg_epoch_num)
    writer.add_scalar("val_cg/r",
                      epoch_val_r_loss / num_val_batch, cg_epoch_num)
    # Auxiliary Loss Functions
    writer.add_scalar("train_cg/link",
                      epoch_train_link_loss/num_train_batch, cg_epoch_num)
    writer.add_scalar("val_cg/link",
                      epoch_val_link_loss/num_val_batch, cg_epoch_num)

    writer.add_scalar("train_cg/ent",
                      epoch_train_ent_loss/num_train_batch, cg_epoch_num)
    writer.add_scalar("val_cg/ent",
                      epoch_val_ent_loss / num_val_batch, cg_epoch_num)

    writer.add_scalar("train_cg/diag",
                      epoch_train_diag_loss/num_train_batch, cg_epoch_num)
    writer.add_scalar("val_cg/diag",
                      epoch_val_diag_loss/num_val_batch, cg_epoch_num)
    total_val_loss = ((epoch_val_r_loss) / num_val_batch)
    cg_model.cg_scheduler.step(total_val_loss)
    return epoch_val_ent_loss / num_val_batch


def _train_u(cg_model, u_model, train_dataloader, val_dataloader, writer,
             u_epoch_num, train_u_only=False):
    """Carries out u training
    """
    num_train_batch = len(train_dataloader)
    num_val_batch = len(val_dataloader)

    epoch_train_total_mse_loss = 0
    epoch_train_total_mae_loss = 0
    epoch_val_total_mse_loss = 0
    epoch_val_total_mae_loss = 0

    epoch_train_force_mse_loss = 0
    epoch_train_force_mae_loss = 0
    epoch_val_force_mse_loss = 0
    epoch_val_force_mae_loss = 0

    epoch_train_energy_mse_loss = 0
    epoch_train_energy_mae_loss = 0
    epoch_val_energy_mse_loss = 0
    epoch_val_energy_mae_loss = 0

    epoch_val_ent_loss = 0

    for (train_x, train_forces, train_energies, batch_size) in train_dataloader:

        train_cg_forces, train_z, ent_loss = cg_model.get_proj_forces_bead(x_data=train_x,
                                                                           x_forces=train_forces,
                                                                           batch_size=batch_size,
                                                                           detach=True)
        mse_force_loss, mae_force_loss, mse_energy_loss, mae_energy_loss, mse_total_loss, mae_total_loss = u_model.get_loss(train_z,
                                                                                                                            train_cg_forces,
                                                                                                                            train_energies)
        u_model.u_optimizer.zero_grad()
        mse_total_loss.backward()
        u_model.u_optimizer.step()
        epoch_train_total_mse_loss += mse_total_loss.item()
        epoch_train_total_mae_loss += mae_total_loss.item()

        epoch_train_force_mse_loss += mse_force_loss.item()
        epoch_train_force_mae_loss += mae_force_loss.item()

        epoch_train_energy_mse_loss += mse_energy_loss.item()
        epoch_train_energy_mae_loss += mae_energy_loss.item()

    for (val_x, val_forces, val_energies, batch_size) in val_dataloader:
        val_cg_forces, val_z, ent_loss = cg_model.get_proj_forces_bead(x_data=val_x,
                                                                       x_forces=val_forces,
                                                                       batch_size=batch_size,
                                                                       detach=True)

        mse_force_loss, mae_force_loss, mse_energy_loss, mae_energy_loss, mse_total_loss, mae_total_loss = u_model.get_loss(val_z,
                                                                                                                            val_cg_forces,
                                                                                                                            val_energies)

        epoch_val_total_mse_loss += mse_total_loss.item()
        epoch_val_total_mae_loss += mae_total_loss.item()

        epoch_val_force_mse_loss += mse_force_loss.item()
        epoch_val_force_mae_loss += mae_force_loss.item()

        epoch_val_energy_mse_loss += mse_energy_loss.item()
        epoch_val_energy_mae_loss += mae_energy_loss.item()
        epoch_val_ent_loss += ent_loss.item()
    if train_u_only:
        u_model.u_scheduler.step(mae_force_loss / num_val_batch)

    writer.add_scalar("train_u/MSETotal",
                      epoch_train_total_mse_loss/num_train_batch, u_epoch_num)
    writer.add_scalar("val_u/MSETotal",
                      epoch_val_total_mse_loss/num_val_batch, u_epoch_num)
    writer.add_scalar("train_u/MAETotal",
                      epoch_train_total_mae_loss/num_train_batch, u_epoch_num)
    writer.add_scalar("val_u/MAETotal",
                      epoch_val_total_mae_loss / num_val_batch, u_epoch_num)

    writer.add_scalar("train_u/MSEForces",
                      epoch_train_force_mse_loss/num_train_batch, u_epoch_num)
    writer.add_scalar("val_u/MSEForces",
                      epoch_val_force_mse_loss/num_val_batch, u_epoch_num)
    writer.add_scalar("train_u/MAEForces",
                      epoch_train_force_mae_loss/num_train_batch, u_epoch_num)
    writer.add_scalar("val_u/MAEForces",
                      epoch_val_force_mae_loss / num_val_batch, u_epoch_num)

    writer.add_scalar("train_u/MSEEnergies",
                      epoch_train_energy_mse_loss/num_train_batch, u_epoch_num)
    writer.add_scalar("val_u/MSEEnergies",
                      epoch_val_energy_mse_loss/num_val_batch, u_epoch_num)
    writer.add_scalar("train_u/MAEEnergies",
                      epoch_train_energy_mae_loss/num_train_batch, u_epoch_num)
    writer.add_scalar("val_u/MAEEnergies",
                      epoch_val_energy_mae_loss / num_val_batch, u_epoch_num)

    return epoch_val_ent_loss / num_val_batch


def train_cg_u(train_dataloader, val_dataloader,
               cg_model, u_model, max_proj_ent=1.0,
               folder_name="./", tag="default", num_epochs=1000,
               u_num_sub_epochs=10, cg_num_sub_epochs=10,
               reg=[0.1, 0.001], freeze_cg_epoch=250,
               add_mean_force_reg=True,
               save_epochs=10, train_u_only=False, detach_link_loss=10000,
               fix_inv_proj_epoch=0):
    """Carries out training of cg and u
    """
    writer = SummaryWriter(log_dir=folder_name + tag)
    train_u_only = train_u_only
    started_u_training = False  # if we started training u
    proj_entropy = -100000

    for epoch in range(num_epochs):
        if not train_u_only:
            for sub_epoch in range(cg_num_sub_epochs):
                cg_epoch_num = epoch * cg_num_sub_epochs + sub_epoch
                proj_entropy = _train_cg(cg_model, u_model, train_dataloader, val_dataloader,
                                         writer, cg_epoch_num, reg, add_mean_force_reg=(started_u_training and add_mean_force_reg),
                                         detach_link_loss=detach_link_loss, fix_inv_proj_epoch=fix_inv_proj_epoch)

                if sub_epoch in [0, cg_num_sub_epochs - 1]:
                    _save_proj_matrices(cg_model, val_dataloader, epoch, root_folder_name=folder_name + tag + "/",
                                        save_tag="cg_" + str(sub_epoch))

        if not train_u_only:
            u_model.reset_u()
        if epoch >= freeze_cg_epoch and proj_entropy < max_proj_ent:
            train_u_only = True



        ###############U TRAINING###################
        if (proj_entropy < max_proj_ent):
            started_u_training = True
            for sub_epoch in range(u_num_sub_epochs):
                u_epoch_num = epoch * u_num_sub_epochs + sub_epoch
                proj_entropy = _train_u(cg_model, u_model, train_dataloader,
                                        val_dataloader, writer, u_epoch_num, train_u_only)
                if sub_epoch in [0]:
                    _save_proj_matrices(cg_model, val_dataloader, epoch, root_folder_name=folder_name + tag + "/",
                                        save_tag="u_" + str(sub_epoch))

        if epoch % save_epochs == 0:
            _save_networks(cg_model, u_model, epoch,
                           root_folder_name=folder_name + tag + "/")
