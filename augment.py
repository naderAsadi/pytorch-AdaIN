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


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/notsketch225_decoder_iter_130000.pth.tar')

# Additional options
parser.add_argument('--content_size', type=int, default=225,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=225,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
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
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])(content)

    return content

def transfer(image, alpha=args.alpha):
    style = style_tf(Image.open(random.choice(style_paths)))
    style = style.to(device).unsqueeze(0)
    with torch.no_grad():
        output = style_transfer(vgg, decoder, image.to(device), style, alpha=alpha)
    output = output.cpu()
    return output

def augment(content, grid_size, img_size):
    n_grids = grid_size ** 2
    tiles = [None] * n_grids
    for n in range(n_grids):
        tiles[n] = get_tile(content, n, grid_size=grid_size)

    # Style whole image
    for i in range(5):
        output_whole = transfer(content, alpha=0.9)
        name = content_path[:-4] + '_trans' + str(i) + content_path[-4:]
        torchvision.utils.save_image(output_whole.data, name)
    
    # Style each tile
    i = 0
    for tile in tiles:
        tile = resize_tile(tile, img_size = img_size).unsqueeze(0)
        for j in range(10):
            output = transfer(tile)
            output = resize_tile(output, img_size=img_size//grid_size)
            name = content_path[:-4] + '_' + str(i) + '_' + str(j) + content_path[-4:]
            torchvision.utils.save_image(output.data, name)
        i += 1
    #tiles = torch.stack(tiles, 0)
    #tiles = torchvision.utils.make_grid(tiles, grid_size, padding=0)
    
    #print(content_path[:-4] + '_1' + args.save_ext)
    #save_image(output, content_path[:-4] + '_1' + args.save_ext)


#################################
# Content & Style Initialization
#################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_paths = [os.path.join(args.content_dir, f) for f in os.listdir(args.content_dir)]
style_paths = [os.path.join(args.style_dir, f) for f in os.listdir(args.style_dir)]

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

if not os.path.exists(args.output):
    os.mkdir(args.output)

##########################
# Networks Initialization
##########################
decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

grid_size = 3
img_size = 225
for content_path in content_paths:
    content = content_tf(Image.open(content_path))
    content = content.unsqueeze(0)

    augment(content, grid_size, img_size)
    
    # print(style.shape)
    # print(content.shape)
    
    # output = output.cpu()
    # output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
    #     args.output, splitext(basename(content_path))[0],
    #     splitext(basename(style_path))[0], args.save_ext
    # )
    # print(content_path[:-4] + '_1' + args.save_ext)
    # save_image(output, content_path[:-4] + '_1' + args.save_ext)
