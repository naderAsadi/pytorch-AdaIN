import argparse
import os
import random
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization
from function import coral

def test_transform(size, crop):
    transform_list = []
    #if size != 0:
    #    transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.Resize([size, size]))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, device, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def get_tile(img, n, grid_size):
    w = int(img.shape[2]) // grid_size
    y = int(n / grid_size)
    x = n % grid_size
    tile = img[:, :, x * w:(x + 1) * w,  y * w:(y + 1) * w]
    return tile

def resize_tile(content, img_size):
    content.squeeze_()
    content = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])(content)

    return content

def transfer(image, style_tf, style_paths, vgg, decoder, device, alpha=1.):
    #style = style_tf(Image.open('./input/style/' + str(i) + '.jpg').convert('RGB'))
    style = style_tf(Image.open(random.choice(style_paths)).convert('RGB'))
    style = style.to(device).unsqueeze(0)
    with torch.no_grad():
        output = style_transfer(vgg, decoder, image.to(device), style, device=device, alpha=alpha)
    output = output.cpu()
    return output

def augment(content, grid_size, img_size, content_path, style_tf, style_paths, vgg, decoder, device, alpha):
    #print(content.shape)
    n_grids = grid_size ** 2
    tiles = [None] * n_grids
    for n in range(n_grids):
        tiles[n] = get_tile(content, n, grid_size=grid_size)

    #Style whole image
    for i in range(5):
        output_whole = transfer(content, style_tf, style_paths, vgg, decoder, device, alpha=alpha)
        name = content_path[:-4] + '_trans' + str(i) + content_path[-4:]
        torchvision.utils.save_image(output_whole.data, name)
    
    # Style each tile
    i = 0
    for tile in tiles:
        #print(tile.shape)
        tile = resize_tile(tile, img_size = img_size).unsqueeze(0)
        for j in range(5):
            output = transfer(tile, style_tf, style_paths, vgg, decoder, device, alpha)
            output = resize_tile(output, img_size=img_size//grid_size)
            name = content_path[:-4] + '_' + str(i) + '_' + str(j) + content_path[-4:]
            torchvision.utils.save_image(output.data, name)
        i += 1

    #print(content_path[:-4] + '_1' + args.save_ext)
    #save_image(output, content_path[:-4] + '_1' + args.save_ext)


def main(content_dir, style_dir, vgg_dir, content_size=225, style_size=225, crop=False, output='output', alpha=1.):
    decoder_dir = 'models/vgg_normalised.pth'
    #vgg_dir = 'models/notart2art_225_2x_decoder_iter_110000.pth.tar'

    #################################
    # Content & Style Initialization
    #################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_paths = [os.path.join(content_dir, f) for f in os.listdir(content_dir)]
    style_paths = [os.path.join(style_dir, f) for f in os.listdir(style_dir)]

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    if not os.path.exists(output):
        os.mkdir(output)

    ##########################
    # Networks Initialization
    ##########################
    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_dir))
    vgg.load_state_dict(torch.load(vgg_dir))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    grid_size = 3
    img_size = 225
    for content_path in content_paths:
        content = content_tf(Image.open(content_path).convert('RGB'))
        content = content.unsqueeze(0)

        augment(content, grid_size, img_size, content_path, style_tf, style_paths, vgg, decoder, device, alpha=alpha)

        # print(style.shape)
        # print(content.shape)

        # output = output.cpu()
        # output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
        #     args.output, splitext(basename(content_path))[0],
        #     splitext(basename(style_path))[0], args.save_ext
        # )
        # print(content_path[:-4] + '_1' + args.save_ext)
        # save_image(output, content_path[:-4] + '_1' + args.save_ext)
