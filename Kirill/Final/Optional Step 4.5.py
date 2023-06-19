from PIL import Image

def make_white_transparent(input_image_path, output_image_path):
    img = Image.open(input_image_path)
    img = img.convert("RGBA")
    
    datas = img.getdata()
    
    new_data = []
    for item in datas:
        # change all white (also shades of whites)
        # pixels to transparent
        if all([x > 200 and x < 256 for x in item[:3]]):
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
            
    img.putdata(new_data)
    img.save(output_image_path, "PNG")

def apply_mask(image_path, mask_path, output_image_path):
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("RGBA")

    image.paste(mask, (0, 0), mask)
    image.save(output_image_path, "PNG")

# Make white transparent for mask.png
make_white_transparent('Graph.png', 'transparent_mask.png')

# Apply transparent_mask.png onto image.jpg
apply_mask('colored_image.png', 'transparent_mask.png', 'masked_image.png')

output_image = Image.open('masked_image.png')
output_image.show()
