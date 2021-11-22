default_square = True
output_size = (299, 299)

# Parameters
# RESNET#####################################################
trained_model = './weights/Resnet50_Final.pth'
network = 'resnet50'

# MOBILENET##################################################
# trained_model = './weights/mobilenet0.25_Final.pth'
# network = 'mobile0.25'

cpu = False

confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
keep_top_k = 750
resize = 1