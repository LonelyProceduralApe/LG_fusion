import os
from PIL import Image

def concatenate_img(images, output_path):
    total_width = 224*8
    max_height = 224
    target_size = (224, 224)

    concatenate_img = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for image in images:
        image = image.resize(target_size, Image.ANTIALIAS)
        concatenate_img.paste(image, (x_offset, 0))
        x_offset += image.width

    concatenate_img.save(output_path)
    print(f"Concatenated {len(images)} images into {output_path}")

def main():
    img_path = r"/home/hello/twelve/ablation/redata/test"

    img_path1 = r"FeatureVisualization"                                                 # 1
    img_path2 = r"/home/hello/zjc/pre/FBSD-main/FeatureVisualization"                   # 2
    img_path3 = r"/home/hello/zjc/pre/WS_DAN_PyTorch-master/FeatureVisualization"       # 3
    img_path4 = r"/home/hello/zjc/pre/Swin-Transformer-main/FeatureVisualization"       # 4
    img_path5 = r"/home/hello/zjc/pre/code/FeatureVisualization"                        # 5
    img_path6 = r"/home/hello/zjc/pre/CAL-master/fgvc/FeatureVisualization"             # 6
    img_path7 = r"/home/hello/zjc/pre/FGVC-PIM-master/FeatureVisualization"             # 7


    output_folder = r"/home/hello/zjc/pre/img_con"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    folder_file_names = os.listdir(img_path)
    for folder_file_name in folder_file_names:
        folder_file_path = os.path.join(img_path, folder_file_name)
        if not os.path.exists(os.path.join(output_folder, folder_file_name)):
            os.makedirs(os.path.join(output_folder, folder_file_name))
        file_names = os.listdir(folder_file_path)
        for file_name in file_names:
            file_name = file_name.split('.')[0]
            images = []

            jpg = os.path.join(img_path, folder_file_name, file_name + '.jpg')
            jpg1 = os.path.join(img_path1, folder_file_name, file_name + '_1.jpg')
            jpg2 = os.path.join(img_path2, folder_file_name, file_name + '_1.jpg')
            jpg3 = os.path.join(img_path3, folder_file_name, file_name + '.jpg')
            jpg4 = os.path.join(img_path4, folder_file_name, file_name + '.jpg')
            jpg5 = os.path.join(img_path5, folder_file_name, file_name + '.jpg')
            jpg6 = os.path.join(img_path6, folder_file_name, file_name + '.jpg')
            jpg7 = os.path.join(img_path7, folder_file_name, file_name + '.jpg')

            images.append(Image.open(jpg))
            images.append(Image.open(jpg1))
            images.append(Image.open(jpg2))
            images.append(Image.open(jpg3))
            images.append(Image.open(jpg4))
            images.append(Image.open(jpg5))
            images.append(Image.open(jpg6))
            images.append(Image.open(jpg7))

            if len(images) == 8:
                output_path = os.path.join(output_folder, folder_file_name, file_name + '.jpg')
                # if not os.path.exists(output_path):
                #     os.makedirs(output_path)
                concatenate_img(images, output_path)

if __name__ == "__main__":
    main()
