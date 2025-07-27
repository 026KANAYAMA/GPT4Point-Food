from collections import defaultdict
import json
import time

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class Ptstext_Benchmark:
    def __init__(self, annotation_file=None):
        # load dataset
        self.dataset,self.anns,self.cats,self.pcds = dict(),dict(),dict(),dict()
        self.pcdToAnns, self.catToPcds = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            # assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, pcds = {}, {}, {}
        pcdToAnns,catToPcds = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                pcdToAnns[ann['pcd_id']].append(ann)
                anns[ann['id']] = ann

        if 'points' in self.dataset:
            for pcd in self.dataset['points']:
                pcds[pcd['id']] = pcd

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToPcds[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns                # for a caption
        self.pcdToAnns = pcdToAnns
        self.catToPcds = catToPcds
        self.pcds = pcds
        self.cats = cats

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = Ptstext_Benchmark()
        # res.dataset['points'] = [pcd for pcd in self.dataset['points']]

        print('Loading and preparing results...')
        with open(resFile) as f:
            anns = json.load(f)

        assert type(anns) == list, 'results in not an array of objects'

         # Cap3D形式のデータを適切に処理
        if isinstance(self.dataset, list):
            # test.jsonの形式：[{"point": "...", "caption": [...]}]
            # 各アイテムからpcd_idを生成してpoint cloudデータを作成
            points_data = []
            annotations_data = []
            
            for i, item in enumerate(self.dataset):
                # point cloudのIDを生成（ファイル名から）
                point_path = item.get('point', '')
                pcd_id = point_path.split('/')[-1].split('.')[0] if point_path else str(i)
                
                # points データを作成
                points_data.append({
                    'id': pcd_id,
                    'file_name': point_path
                })
                
                # annotations データを作成（ground truth用）
                for caption in item.get('caption', []):
                    annotations_data.append({
                        'id': len(annotations_data),
                        'pcd_id': pcd_id,
                        'caption': caption
                    })
            
            res.dataset['points'] = points_data
            res.dataset['annotations'] = annotations_data
        else:
            # 元の形式の場合
            if 'points' in self.dataset:
                res.dataset['points'] = [pcd for pcd in self.dataset['points']]
            else:
                res.dataset['points'] = []

        # 結果ファイルの処理
        processed_anns = []
        for i, ann in enumerate(anns):
            # 結果ファイルのpcd_idを正規化
            if 'pcd_id' in ann:
                processed_anns.append({
                    'id': i + 1,
                    'pcd_id': ann['pcd_id'],
                    'caption': ann.get('caption', '')
                })

        if processed_anns:
            # pcd_idの対応チェック
            result_pcd_ids = set([ann['pcd_id'] for ann in processed_anns])
            gt_pcd_ids = set([pcd['id'] for pcd in res.dataset['points']])
            
            if not result_pcd_ids.issubset(gt_pcd_ids):
                print(f"Warning: Some result pcd_ids not found in ground truth")
                print(f"Result IDs: {result_pcd_ids}")
                print(f"GT IDs: {gt_pcd_ids}")

        res.dataset['annotations'] = processed_anns if processed_anns else anns
        res.createIndex()
        
        # annsPcdIds = [ann['pcd_id'] for ann in anns]
        # assert set(annsPcdIds) == (set(annsPcdIds) & set(self.getPcdIds())), \
        #        'Results do not correspond to current coco set'

        # if 'caption' in anns[0]:
        #     pcdIds = set([pcd['id'] for pcd in res.dataset['points']]) & set([ann['pcd_id'] for ann in anns])
        #     res.dataset['points'] = [pcd for pcd in res.dataset['points'] if pcd['id'] in pcdIds]
        #     for id, ann in enumerate(anns):
        #         ann['id'] = id+1

        # res.dataset['annotations'] = anns
        # res.createIndex()
        return res


    def getPcdIds(self, pcdIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        pcdIds = pcdIds if _isArrayLike(pcdIds) else [pcdIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(pcdIds) == len(catIds) == 0:
            ids = self.pcds.keys()
        else:
            ids = set(pcdIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToPcds[catId])
                else:
                    ids &= set(self.catToPcds[catId])
        return list(ids)