from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp
config = edict()
config.margin_list = (1.0, 0.5, 0.0)   # arcface
config.network = "vit_s_dp005_mask_0"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 128
config.optimizer = "adamw"
config.lr = 0.0001
config.verbose = 2000
config.dali = False

config.rec = "/mnt/workspace/osscmdlibary/glint360"  #glint360k
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 40
config.warmup_epoch = config.num_epoch // 10
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]

