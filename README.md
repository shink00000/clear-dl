# Clear DL - Deep Learning with Clear Code -

In this repository, I will reproduce and implement major models in the field of image recognition and summarize the results of performance measurements.
I will be creating models for the following areas of image recognition.
* object detection
* semantic segmentation
* instance segmentation

# Results
## Detection Models
### [RetinaNet](https://arxiv.org/abs/1708.02002)
* [config](./configs/detection/retinanet_r50_voc_h512_w512.yaml)
    * data: PascalVOC
    * input_size: (512, 512)
    * backbone: ResNet50
* [train log](./results/detection/retinanet_r50_voc_h512_w512/20220213_220417.log)
* [tensorboard](https://tensorboard.dev/experiment/t9GfKu4KRb6L9uoaMZm84Q/)

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

### [FCOS](https://arxiv.org/abs/1904.01355)
* [config](./configs/detection/fcos_r50_voc_h512_w512.yaml)
    * data: PascalVOC
    * input_size: (512, 512)
    * backbone: ResNet50
* [train log](./results/detection/fcos_r50_voc_h512_w512/20220213_014940.log)
* [tensorboard](https://tensorboard.dev/experiment/8FZPdiPcQ2u0mWC9BCkRRQ/)

* evaluation result
    ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.537
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.793
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.584
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.219
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.368
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.598
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.481
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.664
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.669
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.424
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.543
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.718
    ```

### [EfficientDet](https://arxiv.org/abs/1911.09070)
#### EfficientDet-D0
* [config](./configs/detection/efficientdet_d0_voc_h512_w512.yaml)
    * data: PascalVOC
    * input_size: (512, 512)
    * backbone: EfficientNet-B0
* [train log](./results/detection/efficientdet_d0_voc_h512_w512/20220221_061306.log)
* [tensorboard](https://tensorboard.dev/experiment/iX8bElLxTmCNATwxRRZ5gg/)

* evaluation result
    ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.508
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.775
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.561
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.167
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.340
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.574
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.462
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.633
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.636
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.354
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.510
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.690
    ```

#### EfficientDet-D2
* [config](./configs/detection/efficientdet_d2_voc_h512_w512.yaml)
    * data: PascalVOC
    * input_size: (512, 512)
    * backbone: EfficientNet-B2
* [train log](./results/detection/efficientdet_d2_voc_h512_w512/20220218_231837.log)
* [tensorboard](https://tensorboard.dev/experiment/GELXEVWeTySJ0xIerrYDRw/)

* evaluation result
    ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.563
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.819
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.619
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.154
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.631
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.491
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.671
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.674
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.382
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.542
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.729
    ```

#### EfficientDet-D4
* [config](./configs/detection/efficientdet_d4_voc_h512_w512.yaml)
    * data: PascalVOC
    * input_size: (512, 512)
    * backbone: EfficientNet-B2
* [train log](./results/detection/efficientdet_d4_voc_h512_w512/20220219_234845.log)
* [tensorboard](https://tensorboard.dev/experiment/J3O6XDsOT2eZy47u7zWbWg/)

* evaluation result
    ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.584
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.831
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.645
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.223
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.399
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.655
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.503
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.690
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.693
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.405
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.569
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.748
    ```

## SemSeg Models
### [UNet](https://arxiv.org/abs/1505.04597)
* [config](./configs/semseg/unet_m_citys_h512_w1024.yaml)
    * data: CityScapes
    * input_size: (512, 1024)
* [train log](./results/semseg/unet_m_citys_h512_w1024/20220222_131257.log)
* [tensorboard](https://tensorboard.dev/experiment/RVygNz7FQUmva0GBSX0erw/)

* evaluation result
    ```
    road        : 0.9610
    sidewalk    : 0.7411
    building    : 0.8854
    wall        : 0.3637
    fence       : 0.3999
    pole        : 0.5338
    trafficlight: 0.5578
    trafficsign : 0.6601
    vegetation  : 0.9055
    terrain     : 0.5247
    sky         : 0.9236
    person      : 0.7062
    rider       : 0.3835
    car         : 0.9152
    truck       : 0.3006
    bus         : 0.4276
    train       : 0.3538
    motorcycle  : 0.1962
    bicycle     : 0.6706

    mean        : 0.6005
    ```