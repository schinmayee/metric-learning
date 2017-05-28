from PIL import Image

def DefaultImageLoader(path):
    return Image.open(path).convert('RGB')

def Resize(img, size):
    im_size = img.size
    if im_size[0] == im_size[1]:
        new_img = img
    elif im_size[0] > im_size[1]:
        p = (im_size[0] - im_size[1])/2
        new_img = Image.new('RGB', (size, size))
        new_img.paste(img, (p, 0))
    elif im_size[0] < im_size[1]:
        p = (im_size[1] - im_size[0])/2
        new_img = Image.new('RGB', (size, size))
        new_img.paste(img, (0, p))
    new_img = new_img.resize((size, size))
    return new_img
