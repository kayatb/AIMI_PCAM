import dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision


def get_model(model_name, pretrained=False):
    """ Return the model. """
    # TODO: add more models here.
    if model_name == "resnet18":
        return torchvision.models.resnet18(num_classes=2, pretrained=pretrained)
    else:
        raise f"Unknown model name {model_name}."


def matplotlib_imshow(img, one_channel=False):
    """ Helper function for inline image display. """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    plt.show()


if __name__ == '__main__':
    train, val, test = dataset.load_dataset(4)

    dataiter = iter(train)
    images, labels = dataiter.next()

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=False)
    # print('  '.join(classes[labels[j]] for j in range(4)))