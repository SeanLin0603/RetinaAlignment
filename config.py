default_square = True
output_size = (299, 299)
draw_landmark = False

# Parameters
# RESNET#####################################################
pretrain_weight = './weights/Resnet50_Final.pth'
network = 'resnet50'

# MOBILENET##################################################
# pretrain_weight = './weights/mobilenet0.25_Final.pth'
# network = 'mobile0.25'

# cpu = False
gpu = True

inner_padding_factor = 0.05
outer_padding = (0, 0)

confidence = 0.6
confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
keep_top_k = 750
resize = 1