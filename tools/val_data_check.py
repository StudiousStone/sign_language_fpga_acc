from class_tools import extract_data
import caffe
import numpy as np

def set_image_as_input(net,image_file):
    
    mean = np.array([97.2227935791,105.6512146,118.904281616],'f')

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mean)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    
    image = caffe.io.load_image(image_file)
    transformed_image = transformer.preprocess('data', image)

    net.blobs['data'].data[...] = transformed_image 

def get_input_with_classification():
    val_set_dir = "/home/localhost/pa1850co-s/Desktop/BoC/caffe_training/ValData/"
    base_info_dir = "/home/localhost/pa1850co-s/tests-caffe/classify-image/"
    output_dir = base_info_dir + "goldendata/inputs/"
    model_dir = base_info_dir + "model/homogeneous_fire_modules/"

    model_file = model_dir + "deploy.prototxt"
    weights_file = model_dir + "snapshot_iter_1296.caffemodel"

    val_txt = val_set_dir + "val.txt"

    net = caffe.Net(model_file,weights_file,caffe.TEST)

    labels_file = output_dir + "labels.txt"
    index_file = output_dir + "index.txt"
    image_file = ""
    image_out_file = ""
    image_label = ""
    line_list = []

    #extract image name from txt and add it to index.txt, also extract the ground truth value
    fd_val_txt = open(val_txt,'r')
    fd_labels = open(labels_file,'w')
    fd_index = open(index_file,'w')

    for line in fd_val_txt:
        line = line[:-1]
        line_list = line.split()

        image_file = val_set_dir + line_list[0]
        line_list[0] = line_list[0].replace('png','txt')
        line_list[0] = line_list[0].replace('jpg','txt')
        image_out_file = output_dir + line_list[0]
        image_label = line_list[1]

        fd_index.write(image_out_file + '\n')
        fd_labels.write(image_label + ',')
        #set as input for network
        set_image_as_input(net,image_file)

        #print image to file
        extract_data.print_data(net.blobs["data"].data[...],image_out_file,False,8,0)

        
    fd_val_txt.close()
    fd_labels.close()
    fd_index.close()

if __name__ == '__main__':
    get_input_with_classification()
