_base_ = '/home/summer23_intern1/workspace/p1/mmdetection/configs/_base_/models/cascade-rcnn_r50_fpn_tooth_detection.py'

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/tooth_detection/'
metainfo = {
    'classes': ('Teeth defect', 'Tooth num 21', 'Tooth num 22', 'Tooth num 23', 
                'Tooth num 24', 'Tooth num 25', 'Tooth num 26', 'Tooth num 27', 
                'Tooth num 28', 'Tooth num 31', 'Tooth num 32', 'Tooth num 11', 
                'Tooth num 33', 'Tooth num 34', 'Tooth num 35', 'Tooth num 36', 
                'Tooth num 37', 'Tooth num 38', 'Tooth num 41', 'Tooth num 42', 
                'Tooth num 43', 'Tooth num 44', 'Tooth num 12', 'Tooth num 45', 
                'Tooth num 46', 'Tooth num 47', 'Tooth num 48', 'Phase 1', 'Phase 2', 'Phase 3', 
                'Amalgam', 'Gold inlay', 'Tooth root exposure', 'Tooth num 13', 
                'Gum inflammation', 'Metal-seramic', 'Gold', 'Metal', 'Tooth num 14', 
                'Tooth num 15', 'Tooth num 16', 'Tooth num 17', 'Tooth num 18'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), 
                (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30), 
                (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), 
                (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255), 
                (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255), (199, 100, 0), 
                (72, 0, 118), (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
                  (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255), 
                  (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255), 
                  (134, 134, 103), (145, 148, 174), (255, 208, 186), (197, 226, 255), 
                  (171, 134, 1), (109, 63, 54), (207, 138, 255)]
}

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
# dataloaders
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='sample/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/valid.json',
        data_prefix=dict(img='sample/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
# test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/valid.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
# test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root+'annotations/test2.json',
        data_prefix=dict(img='sample/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'proposal'],
    # format_only=True,
    format_only=False,
    ann_file=data_root + 'annotations/test2.json',
    outfile_prefix='./work_dirs/tooth_detection/test',
    backend_args=backend_args,
    classwise=True,
    )

# scheduler settings 

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


# Do not omit anything from the code below

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend'), 
                dict(type='WandbVisBackend',
                     init_kwargs=dict(project='tooth-detection-20e'))]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')


log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
work_dir = 'work_dirs/tooth_detection_20e'
log_level = 'INFO'
load_from = None
resume = False
