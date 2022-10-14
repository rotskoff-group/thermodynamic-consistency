from torch.utils.tensorboard import SummaryWriter

def train_forward_loss(nf_model, ic_dataloader,
                       num_epochs=1000, folder_name = "./", tag="default"):
    """Trains normalizing flow with a forward loss
    """
    writer = SummaryWriter(log_dir=folder_name + tag)
    num_batches = len(ic_dataloader)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for ic_batch in ic_dataloader:
            forward_loss = nf_model.compute_forward_loss_from_dataset(ic_batch)
            nf_model.flow_optimizer.zero_grad()
            forward_loss.backward()
            nf_model.flow_optimizer.step()
            epoch_loss += forward_loss.item()

        nf_model.flow_scheduler.step(epoch_loss/num_batches)


        writer.add_scalar("Forward Loss", epoch_loss/num_batches, epoch)
        if epoch % 25 == 0:
            nf_model.save(epoch_num=epoch)