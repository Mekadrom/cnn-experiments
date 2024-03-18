import os
import torch
import torchvision
import trainer
import vae.utils as utils

class NormalTrainer(trainer.Trainer):
    def __init__(self, args, train_dataloader_provider, test_dataloader_provider, vae):
        super(NormalTrainer, self).__init__(args, train_dataloader_provider, test_dataloader_provider, vae)

    def train(self):
        torch.save({'epoch': -1, 'model': self.vae, 'args': str(self.args)}, os.path.join("runs", self.args.run_name, f"model_none.pt"))
        train_dataloader = self.get_data_loaders('train')
        test_dataloader = self.get_data_loaders('test')

        if hasattr(train_dataloader, 'dataset'):
            test_image = test_dataloader.dataset[0][-1].to(self.args.device)
        elif hasattr(train_dataloader, 'data'):
            test_image = test_dataloader.data[-1].to(self.args.device)
        else:
            raise ValueError("Unknown dataloader type")
        
        torchvision.utils.save_image(self.get_image_example(None, test_image), os.path.join("runs", self.args.run_name, "test_image_uninitialized.png"))

        for epoch in range(self.args.n_epochs):
            if hasattr(train_dataloader, 'create_batches'):
                train_dataloader.create_batches()

            train_losses = self.train_epoch(epoch, train_dataloader)
            torch.save({'epoch': epoch, 'model': self.vae.state_dict(), 'optim': self.vae_optimizer}, os.path.join("runs", self.args.run_name, f"model.pt"))

            self.summary_writer.add_image("test_reconstructed_image", self.get_image_example(train_dataloader), self.train_steps)

            if hasattr(test_dataloader, 'create_batches'):
                test_dataloader.create_batches()
                
            val_losses = self.validate_epoch(test_dataloader)

            print(f"Epoch {epoch} - Train loss: {train_losses} - Validation loss: {val_losses}")

            self.summary_writer.add_scalar("validation_loss", val_losses, epoch)

            # save image to file
            torchvision.utils.save_image(self.get_image_example(None, test_image), os.path.join("runs", self.args.run_name, f"test_image_{epoch}.png"))

    def train_epoch(self, epoch, dataloader):
        self.vae.train()

        kl_losses = utils.AverageMeter()
        recon_losses = utils.AverageMeter()
        losses = utils.AverageMeter()

        batch_count = 0
        for i, (data, _) in enumerate(dataloader):
            x = data.to(self.args.device)
            self.vae_optimizer.zero_grad()

            x_reconstructed, mu, log_var = self.vae(x)

            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            recon_loss = self.vae_criterion(x_reconstructed, x)

            loss = kl_loss + recon_loss
            (loss / self.args.batches_to_accumulate).backward()
            
            kl_losses.update(kl_loss.item(), x.size(0))
            recon_losses.update(recon_loss.item(), x.size(0))
            losses.update(loss.item(), x.size(0))

            if batch_count % self.args.batches_to_accumulate == 0:
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)

                self.vae_optimizer.step()

                self.train_steps += 1

                print('Epoch {0}/{1}-----Batch {2}/{3}-----Model update #{4}-----KL Loss {kl_losses.val:.4f} ({kl_losses.avg:.4f})-----Recon Loss {recon_losses.val:.4f} ({recon_losses.avg:.4f})-----Loss {total_losses.val:.4f} ({total_losses.avg:.4f})'
                      .format(epoch + 1, self.args.n_epochs, i, len(dataloader), self.train_steps, kl_losses=kl_losses, recon_losses=recon_losses, total_losses=losses))
                
                self.summary_writer.add_scalar("train_loss", loss.item(), self.train_steps)
                # self.summary_writer.add_image("train_reconstructed_image", self.get_image_example(dataloader), self.train_steps)

            if i % self.args.test_image_steps == 0:
                self.summary_writer.add_image("train_reconstructed_image", self.get_image_example(dataloader), self.train_steps)
                self.vae.train()

            batch_count += 1

        return losses.avg

    def validate_epoch(self, dataloader):
        self.vae.eval()

        losses = utils.AverageMeter()

        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                x = data.to(self.args.device)
                
                x_reconstructed, mu, log_var = self.vae(x)

                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                recon_loss = self.vae_criterion(x_reconstructed, x)

                loss = torch.mean(kl_loss + recon_loss)

                losses.update(loss.item(), x.size(0))

            print(f"Validation loss: {losses.avg}")

        return losses.avg
