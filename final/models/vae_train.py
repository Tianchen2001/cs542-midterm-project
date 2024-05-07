import matplotlib.pyplot as plt

from models.utils import show_images, deprocess_img, preprocess_img


def train_vae(
    VAE,
    optimizer,
    loss_function,
    show_every=250,
    batch_size=128,
    num_epochs=10,
    train_loader=None,
    device=None,
):
    iter_count = 0
    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch + 1}")
        for x, _ in train_loader:
            _, input_channels, img_size, _ = x.shape
            
            x = x.to(device)
            
            # Forward pass
            recon_x, mu, logvar = VAE(x)
            
            # Calculate loss
            loss = loss_function(recon_x, x, mu, logvar)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging and output visualization
            if iter_count % show_every == 0:
                print(
                    "Iter: {}, loss: {:.4}".format(
                        iter_count, loss.item()
                    )
                )
                disp_fake_images = recon_x.data
                imgs_numpy = (disp_fake_images).cpu().numpy()
                if input_channels != 1:
                    imgs_numpy = imgs_numpy.reshape(-1, input_channels, img_size, img_size)
                show_images(imgs_numpy[0:16], color=input_channels != 1)
                plt.show()
                print()
            iter_count += 1