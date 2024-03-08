import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
def define_rpn(AA):
    min_size=800
    max_size=1333
    image_mean=None
    image_std=None,
    # RPN parameters
    rpn_anchor_generator=None
    rpn_head=None
    rpn_pre_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=1000
    rpn_post_nms_top_n_train=2000
    rpn_post_nms_top_n_test=1000
    rpn_nms_thresh=0.7
    rpn_fg_iou_thresh=0.7
    rpn_bg_iou_thresh=0.3
    rpn_batch_size_per_image=256
    rpn_positive_fraction=0.5
    rpn_score_thresh=0.0
    # Box parameters
    box_roi_pool=None
    box_head=None
    box_predictor=None
    box_score_thresh=0.05
    box_nms_thresh=0.5
    box_detections_per_img=100
    box_fg_iou_thresh=0.5
    box_bg_iou_thresh=0.5
    box_batch_size_per_image=512
    box_positive_fraction=0.25
    bbox_reg_weights=None

    ########################## anchor生成器
    anchor_sizes = ((AA,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    print(aspect_ratios)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )


    #########################RPN头
    #out_channels 为骨干网络输出的通道数
    out_channels = 1024
    print(rpn_anchor_generator.num_anchors_per_location())
    rpn_head = RPNHead(
        out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
    )

    rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
    rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
    rpn = RegionProposalNetwork(
                rpn_anchor_generator, rpn_head,
                rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                rpn_batch_size_per_image, rpn_positive_fraction,
                rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
                )
    # images = []
    # image1 = torch.randn(2,3,256,256)
    # images = ImageList(image1,[[256,256],[256,256],[256,256]])
    # p1,p2 = rpn(images,d)
    return rpn;
