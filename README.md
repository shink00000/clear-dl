# Clear DL - Deep Learning with Clear Code -

In this repository, I will reproduce and implement major models in the field of image recognition and summarize the results of performance measurements.
I will be creating models for the following areas of image recognition.
* object detection
* semantic segmentation
* instance segmentation

# Results
## Detection Models
### [RetinaNet](https://arxiv.org/abs/1708.02002)
* [config](./configs/detection/retinanet_r50_voc_h512_w512.py)
    * data: PascalVOC
    * input_size: (512, 512)
    * backbone: ResNet50
* [train log](./results/retinanet_r50_voc_h512_w512/20220213_220417.log)
* [tensorboard](https://tensorboard.dev/experiment/BnK35eDhRqGMUZvRy0PheA/#scalars)

* evaluation result
    ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.562
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.824
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.623
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.212
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.406
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.621
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.488
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.669
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.672
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.413
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.560
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.720
    ```
