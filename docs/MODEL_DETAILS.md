# MODEL DETAILS
In paper SpineNet was trained with three protocols. I only trained with protocol B with 8 Tesla V100 gpus.

- protocol  A: random scale between[0.8,1.2];250 epochs;DropBlock
- protocol  B: random scale between[0.5,2.0];350 epochs
- protocol  C: random scale between[0.5,2.0];500 epochs;stochastic depth;Swish

### mAP

| Backbone     | Resolution | my code(B) | paper(A) | paper(B) | paper(C) |Training Time|
| ------------ | ---------- | ---------- | -------- | -------- | -------- | ----------- |
| SpineNet-49S |  640x640   | 39.2       | ——       | 39.9     | 41.5     |   3d 5h     |
| SpineNet-49  |  640x640   | 42.1       | 40.8     | 42.8     | 44.3     |   4d 2h     |
| SpineNet-49  |  896x896   | 44.9       | ——       | 45.3     | 46.7     |   6d 16h    |
| SpineNet-96  |  1024x1024 | 46.9       | ——       | 47.1     | 48.6     |   11d 7h    |

### Details

- SpineNet-49S
    ```
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.392
	 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.587
	 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.419
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.209
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.424
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.556
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.326
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.520
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.552
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.332
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.599
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.732
    ```
- SpineNet-49 640
    ```
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
	 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.621
	 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.452
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.242
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.581
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.342
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.545
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.576
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.372
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.625
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.746
    ```
- SpineNet-49 896
	```
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.449
	 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.652
	 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.484
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.274
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.492
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.599
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.355
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.574
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.609
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.428
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.650
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.756
	 ```
- SpineNet-96 1024
	```
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.469
	 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.673
	 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.507
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.300
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.510
	 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.608
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.364
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.588
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.624
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.448
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.665
	 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.768
	 ```




