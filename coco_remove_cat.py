import json
import argparse
import copy 


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--newfile', type=str)
parser.add_argument('--classes', nargs='+')
args = parser.parse_args()

def save_coco(file, info, images, annotations, categories):
    with open(file, 'w') as coco:
        json.dump({ 'info': info, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)
        
def main(args):
    with open(args.file, 'r') as rf:
        coco = json.load(rf)   
        info = coco['info']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        new_categories = copy.deepcopy(categories)
        # new_annotations = copy.deepcopy(annotations)
        new_annotations = []

        classes = args.classes

        for cat in categories:
            if str(cat['id']) in classes: 
                new_categories.remove(cat)

        for ann in annotations:
            if str(ann['category_id']) in classes: 
                continue 
            new_annotations.append(ann)

        save_coco(args.newfile, info, images, new_annotations, new_categories)

if __name__ == "__main__":
    main(args) 