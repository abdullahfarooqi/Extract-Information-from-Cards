from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
import imageio
import pandas as pd
import re
import glob
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            try:
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
            except:
                pass
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

# apply the function to convert all XML files in images/ folder into labels.csv
labels_df = xml_to_csv('Anotated-UK Passport-YOLO/')
labels_df.to_csv(('labels4.csv'), index=None)
def bbs_obj_to_df(bbs_object):
    bbs_array = bbs_object.to_xyxy_array()
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs

#Augmentation

aug = iaa.SomeOf(1,[
    iaa.Affine(rotate=(0, 4))
])

def image_aug(df, images_path, aug_images_path, image_prefix, augmentor):
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
                              )
    grouped = df.groupby('filename')

    for filename in df['filename'].unique():
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)
        #   read the image
        image = imageio.imread(images_path + filename)
        bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image()
        bbs_aug = bbs_aug.clip_out_of_image()
        if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
            pass
        else:
            imageio.imwrite(aug_images_path + image_prefix + filename, image_aug)
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix + x)
            bbs_df = bbs_obj_to_df(bbs_aug)
            aug_df = pd.concat([info_df, bbs_df], axis=1)
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])

    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy

augmented_images_df = image_aug(labels_df, 'Images3/', 'aug_images/', 'aug1_', aug)
all_labels_df = pd.concat([augmented_images_df])
all_labels_df.to_csv('all_labels.csv', index=False)