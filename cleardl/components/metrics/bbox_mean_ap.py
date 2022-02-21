from torchmetrics import Metric
from os import devnull
from io import StringIO
from contextlib import redirect_stdout
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class BBoxMeanAP(Metric):
    """
    Inputs:
        preds: Tuple[List[Tensor], List[Tensor], List[Tensor]]
            - [bboxes_batch1, bboxes_batch2, ...]
            - [scores_batch1, scores_batch2, ...]
            - [class_ids_batch1, class_ids_batch2, ...]
        metas: Tuple[Dict[str, Any]]
            contains 'image_id'

    Outputs:
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = x.xxx
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = x.xxx
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = x.xxx
    """

    def __init__(self, anno_path: str, pred_size: list, classwise: bool = False):
        super().__init__()

        with redirect_stdout(open(devnull, 'w')):
            self.cocoGt = COCO(anno_path)
        self.ph, self.pw = pred_size
        self.classwise = classwise
        self.add_state('coco_dets', default=[], dist_reduce_fx=None)

    def update(self, preds: tuple, metas: tuple):
        pred_bboxes, pred_scores, pred_class_ids = preds

        for batch_id in range(len(metas)):
            bboxes = pred_bboxes[batch_id].cpu().numpy()
            scores = pred_scores[batch_id].cpu().numpy()
            class_ids = pred_class_ids[batch_id].cpu().numpy()
            meta = metas[batch_id]
            h_ratio = meta['height'] / self.ph
            w_ratio = meta['width'] / self.pw
            dets = [{
                'image_id': meta['image_id'],
                'category_id': class_id,
                'bbox': [
                    xmin * w_ratio,
                    ymin * h_ratio,
                    (xmax - xmin) * w_ratio,
                    (ymax - ymin) * h_ratio],
                'score': score
            } for (xmin, ymin, xmax, ymax), score, class_id in zip(bboxes, scores, class_ids)]
            self.coco_dets.extend(dets)

    def compute(self) -> dict:
        buf = StringIO()
        with redirect_stdout(open(devnull, 'w')):
            cocoGt = self.cocoGt
            cocoDt = cocoGt.loadRes(self.coco_dets)
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            if self.classwise:
                for cat_id in cocoGt.getCatIds():
                    cocoEval.params.catIds = [cat_id]
                    cocoEval.evaluate()
                    cocoEval.accumulate()
                    with redirect_stdout(buf):
                        print(cocoGt.cats[cat_id]['name'] + ':')
                        cocoEval.summarize()
            else:
                cocoEval.params.catIds = cocoGt.getCatIds()
                cocoEval.evaluate()
                cocoEval.accumulate()
                with redirect_stdout(buf):
                    cocoEval.summarize()
        return {
            'text': buf.getvalue(),
            '@IoU=0.50:0.95': cocoEval.stats[0],
            '@IoU=0.50': cocoEval.stats[1]
        }
